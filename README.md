# jobdash — always-on job dashboard

A local web service that watches your Slurm jobs on any cluster you can ssh to,
plus tmux jobs on a workstation with no scheduler, keeps a real time series in
SQLite, and serves an interactive dashboard on `http://localhost:8787`.

It is read-only. No route in this app can cancel, hold, or requeue a job.

Every route reads SQLite or the poller's in-memory snapshot, so page loads never
wait on ssh. `/api/logtail` is the one exception. It runs its own ssh on a worker
thread when you click "fetch log tail", so a slow cluster delays that request and
nothing else.

## Layout

| File | Role |
|---|---|
| `collector.py` | The original `jobdash.py`. One SSH round-trip per machine per refresh, and the `kind` dispatch that picks a backend. Still runs standalone as a terminal dashboard. |
| `tmuxhost.py` | The workstation backend. Turns tmux panes into the same `Job`s Slurm produces. |
| `store.py` | SQLite schema and writes. |
| `queries.py` | Read-side aggregations for the UI. |
| `tests/` | Aggregation tests against an in-memory SQLite fixture. |
| `alerts.py` | Alert rules, keyed per job so state can persist. |
| `poller.py` | Background thread. The only thing that talks to the machines. |
| `app.py` | FastAPI routes and the SSE stream. |
| `static/`, `vendor/` | Frontend. No build step; uPlot is vendored, not hotlinked. |

## Requirements

| Need | Why |
|---|---|
| Python 3.11+ | `tomllib`, for the config file |
| An ssh alias per machine, in `~/.ssh/config` | every backend is one ssh round-trip |
| Key-based ssh with no passphrase prompt | the poller runs `BatchMode=yes`, so a prompt is a failure, not a question |
| macOS, only for the login-at-boot service and desktop notifications | the app itself runs anywhere |

Check ssh before anything else. Each host must answer without asking you for
anything:

```bash
ssh gaivi true && echo ok
```

If that hangs or prompts, fix `~/.ssh/config` and `ssh-copy-id` first. jobdash
cannot work around it.

## Install

```bash
git clone <repo-url> jobdash
cd jobdash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Write a starter config and edit it. One `[[cluster]]` block per machine, where
`host` is the ssh alias and `user` is your account on that machine:

```bash
.venv/bin/python collector.py --write-config
$EDITOR ~/.config/jobdash/config.toml
```

Then check that each cluster answers. `--check` diagnoses one step at a time and
tells you which one broke:

```bash
.venv/bin/python collector.py --check
```

## Run it

```bash
.venv/bin/python app.py            # then open http://localhost:8787
```

The database is created on first run as `jobdash.db` next to `app.py`. Set
`JOBDASH_DB` to put it somewhere else, and `JOBDASH_CONFIG` to read the config
from somewhere other than `~/.config/jobdash/config.toml`.

Terminal mode still works, unchanged:

```bash
.venv/bin/python collector.py --once
.venv/bin/python collector.py --check
```

## Install as a service (macOS)

`install-service.sh` fills your own paths into `com.jobdash.plist.template`,
writes the result to `~/Library/LaunchAgents`, and starts it. Run it from the
project directory, after the venv exists:

```bash
./install-service.sh
```

Pass a port to use something other than 8787 (`./install-service.sh 9000`).
Re-running it replaces the loaded agent, so it is also how you apply a change.

Logs land in `~/Library/Logs/jobdash.log` and `.err`.

Two things to know. A LaunchAgent only runs while the user is logged in, so a
headless Mac mini wants auto-login enabled. Do not convert this to a
LaunchDaemon — it would run as root, which does not have your SSH keys.

On Linux, skip this and write a systemd user unit that runs the same
`.venv/bin/python app.py` command, or just run it under tmux.

## Reaching it from another machine

The service binds `127.0.0.1` on purpose. Tunnel in rather than opening a port:

```bash
ssh -N -L 8787:localhost:8787 macmini
```

## The tabs

| Tab | What it answers |
|---|---|
| overview | What is running right now. Filter by name, jobid, partition, or node; the loss column carries a sparkline so trend shows without a click. |
| usage | Where the GPU-hours went, what fraction ran idle, how long the queue makes you wait, and how much memory each finished job asked for against what it peaked at. |
| clusters | How busy each machine is for everyone, not just you. Queue depth and free GPUs per partition, node states, the GPU model inventory, live wattage and the energy bill. |
| compare | Loss curves for several runs on one step axis. Runs of different lengths and start times line up, which is what reading a sweep needs. |
| history | The Gantt strip and the finished-job table. |
| alerts | Every alert in the last 14 days, acked or not. |

The tab, the selected job, and the filter live in the URL hash, so a reload
lands where you left off and a job view can be pasted to someone else.

## The workstation

A machine with no scheduler needs `kind = "tmux"`, which picks a different
backend for it. The examples below use my workstation, `wrx80`.
Every tmux pane holding a live process is a job; a pane sitting at a prompt is
not. The rest of the app never branches on this, because `tmuxhost.py` returns
the same `Snapshot` the Slurm path does.

| Slurm | on a tmux host |
|---|---|
| job id | `<pid>_<start epoch>`, so two runs that reuse a pid stay separate |
| squeue state | always `R` — a pane either holds a process or does not |
| partition | the tmux session name |
| node | `session:window.pane`, plus the cards it holds (`jdgpu:0.0 gpu1`) |
| walltime left | `UNLIMITED`, so the WALL alert never fires here |
| StdOut | the job's fd 1 when that is a file, else the pane's scrollback |
| sacct | nothing — `store.close_vanished` writes the end when a job disappears |

The job is the shallowest non-shell process at or under the pane, so
`tmux new-window python train.py` and a `python train.py` typed at a prompt both
resolve to the same process. GPU memory is attributed across that process's
whole tree, which is what catches dataloader workers and `torchrun` children.

A run started as `python train.py > run.log` gets a real log file out of fd 1,
which means an exact mtime, an error scan, and a full-history metric grep —
everything the Slurm path gets from `StdOut`. A run printing straight to the
terminal falls back to the pane buffer, capped at the last 2000 lines.

Name the tmux window after the run (`tmux new-window -n sweep-a`) and that name
becomes the job name. Otherwise it falls back to the script being run, since
every job would otherwise be called "python3".

## Cluster-wide stats

The per-job view runs `squeue -u <you>`, which cannot say whether a slow queue
is the cluster being full or your submission being wrong. Two more commands ride
the same round-trip and answer that:

| Command | What it gives |
|---|---|
| `squeue -h -o '%T\|%P' \| sort \| uniq -c` | running / pending / other, per partition, aggregated remotely |
| `sinfo -h -O 'Partition,StateLong,Nodes,CPUsState,Gres,GresUsed'` | node states, CPU alloc/idle, and GPUs configured vs allocated with the model name |

Both are aggregated on the remote side. Shipping raw `squeue` for a busy cluster
every 20 s would undo the one-round-trip budget the collector is built around.

`sinfo` emits one row per partition+state pair, so a node in three partitions is
counted in each. The whole-cluster `ALL` row is therefore built from a separate
per-node view deduped on node name, not by summing partitions.

## Power and energy

`nvidia-smi` reports `power.draw`, so the GPU probe reads real watts rather than
guessing. Board TDP x utilization is the fallback for cards that answer `[N/A]`,
and the two are stored in separate columns and shown apart, because the estimate
is bad in both directions — a card at 2% util still pulled 260 W on gaivi, and a
Blackwell at 66% pulled 292 W against a 1200 W estimate.

Energy integrates watts over the gap between samples with the same `GAP_CAP_S`
cap `gpu_hours` uses, so an outage is not billed. `price_per_kwh` under
`[global]` turns kWh into dollars.

The srun probe is scoped to the job's own cards when the cluster sets
`ConstrainDevices=yes` — gaivi does. The ssh-to-node fallback runs outside the
job step and sees the whole box, so those samples are tagged `gpu_scope='node'`
and the reading is flagged in the UI as an upper bound.

## Configuration

`~/.config/jobdash/config.toml`, written by `collector.py --write-config`.
`interval` sets the SSH cadence, and the
browser follows it over SSE rather than polling on its own clock.
`price_per_kwh` sets the electricity rate the energy panel prices against.

Set `notify_url` under `[global]` to send new alerts somewhere other than the
Mac's notification centre. A Slack or Discord webhook URL is posted as JSON;
anything else — an ntfy topic, say — gets plain text. The POST runs on its own
thread, so an unreachable endpoint cannot stretch the refresh.

```toml
[global]
notify = true
notify_url = "https://ntfy.sh/your-topic"
```

A machine without a scheduler needs `kind = "tmux"`. Everything else in a
`[[cluster]]` block means the same thing either way.

```toml
[[cluster]]
name = "wrx80"
host = "wrx80"
user = "justin"
kind = "tmux"
```

## Stopping it

Once loaded, `KeepAlive` means launchd restarts the service the moment it exits,
so `pkill` will not stop it.

```bash
launchctl bootout gui/$UID/com.jobdash
```

## Tests

The aggregations in `queries.py` are the easiest thing here to get subtly wrong,
so they are pinned against a seeded in-memory database. `test_tmuxhost.py` pins
the workstation field mapping against verbatim remote output, so a change to the
remote script's shape breaks a test rather than the dashboard.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

## Retention

Raw samples land at one row per job per poll. Anything older than three days is
folded into `samples_hourly` — one row per job per hour, carrying the same
capped-gap `dt_s` the raw integration uses, so GPU-hours match across the
boundary. `prune` drops both tables past 90 days. The poller runs rollup and
prune once a day; startup no longer owns the only chance to do housekeeping.

`cluster_samples` follows the same three-day boundary but rolls up in place,
since there is no per-job dimension to collapse. Counts average rather than sum
— queue depth is a level, not an event — and `n` records how many samples went
into a row, so a chart can tell an instantaneous count from an hourly mean.

## Known gaps

The cluster names below are my own — gaivi and iruka2 are Slurm clusters,
wrx80 is a workstation. The behaviour they illustrate is not specific to them.

- **A shared GPU is billed to both jobs.** nvidia-smi reports utilization per
  device, not per process, so two runs on one card each read the card's full
  utilization and `gpu_hours` counts that hour twice. Slurm's exclusive GRES
  made this impossible on the clusters; a workstation does not, and there is
  nothing in the driver to divide it with. Per-process GPU *memory* is exact.

- **Workstation jobs end as `ENDED`, never COMPLETED or FAILED.** A foreground
  command in a pane leaves no exit code behind to read, so the finished row
  records that the job stopped and nothing more. The `errors` column and the
  ERR alert still carry whether it went wrong. `failures` grows an `ENDED`
  bucket because of this.

- **DIVERGE is weaker on a job with no log file.** The rule compares the last
  loss against the median of the run's early history. With a log file that is
  the real start of the run; with only the pane buffer it is whatever the
  oldest of 2000 scrollback lines happens to be, so the alert rarely fires and
  can fire spuriously if the buffer begins mid-warmup. Redirect a run to a file
  if the divergence alert matters for it.

- **iruka2 reports no GPU utilization.** Both probes are ruled out. `srun
  --overlap` still times out waiting for the job step, and the compute nodes
  refuse a direct ssh (`Permission denied (publickey,password)`), so the
  node-side `nvidia-smi` fallback cannot run either. `gpu = false` stays set for
  it in `config.toml`, and the utilization chart, GPU-hours, and efficiency
  numbers remain gaivi only. `scontrol show job -d` does report the allocated
  GPU index (`GRES=gpu:...(IDX:3)`), so if node access is ever granted, that is
  the way in. Cluster-wide queue and node stats still work there, since they come
  from `squeue` and `sinfo` rather than from the cards.

- **MaxRSS needs its own sacct query.** The history query runs `sacct -X`, which
  returns allocation lines only, and MaxRSS lives on the `.batch` step. A second
  narrow query (`-o JobID,ReqMem,MaxRSS`, srun steps grepped out) supplies both
  halves, joined on the base job id. Jobs that finished before that query existed
  have no memory row and are left out of the panel rather than shown at 0%.
