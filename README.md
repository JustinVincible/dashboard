# jobdash — always-on Slurm dashboard

A local web service that watches your Slurm jobs on gaivi and iruka2, keeps a
real time series in SQLite, and serves an interactive dashboard on
`http://localhost:8787`.

It is read-only. No route in this app can cancel, hold, or requeue a job.

Every route reads SQLite or the poller's in-memory snapshot, so page loads never
wait on ssh. `/api/logtail` is the one exception. It runs its own ssh on a worker
thread when you click "fetch log tail", so a slow cluster delays that request and
nothing else.

## Layout

| File | Role |
|---|---|
| `collector.py` | The original `jobdash.py`. One SSH round-trip per cluster per refresh. Still runs standalone as a terminal dashboard. |
| `store.py` | SQLite schema and writes. |
| `queries.py` | Read-side aggregations for the UI. |
| `tests/` | Aggregation tests against an in-memory SQLite fixture. |
| `alerts.py` | Alert rules, keyed per job so state can persist. |
| `poller.py` | Background thread. The only thing that talks to the clusters. |
| `app.py` | FastAPI routes and the SSE stream. |
| `static/`, `vendor/` | Frontend. No build step; uPlot is vendored, not hotlinked. |

## Run it

```bash
.venv/bin/python app.py            # then open http://localhost:8787
```

Terminal mode still works, unchanged:

```bash
.venv/bin/python collector.py --once
.venv/bin/python collector.py --check
```

## Install as a service

Run these from the `Dashboard` directory. `~/Library/LaunchAgents` does not
exist on a fresh account, and `launchctl load` reports a missing plist as
"Input/output error", so create the directory first and use `bootstrap`, which
gives real error messages.

```bash
mkdir -p ~/Library/LaunchAgents
cp com.jmcmillen.jobdash.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.jmcmillen.jobdash.plist
launchctl print gui/$UID/com.jmcmillen.jobdash | grep -E "state|pid|last exit"
```

Logs land in `~/Library/Logs/jobdash.log` and `.err`.

Two things to know. A LaunchAgent only runs while the user is logged in, so a
headless Mac mini wants auto-login enabled. Do not convert this to a
LaunchDaemon — it would run as root, which does not have your SSH keys.

## Reaching it from another machine

The service binds `127.0.0.1` on purpose. Tunnel in rather than opening a port:

```bash
ssh -N -L 8787:localhost:8787 macmini
```

## The tabs

| Tab | What it answers |
|---|---|
| overview | What is running right now. Filter by name, jobid, partition, or node; the loss column carries a sparkline so trend shows without a click. |
| usage | Where the GPU-hours went, what fraction ran idle, and how long the queue makes you wait. |
| compare | Loss curves for several runs on one step axis. Runs of different lengths and start times line up, which is what reading a sweep needs. |
| history | The Gantt strip and the finished-job table. |
| alerts | Every alert in the last 14 days, acked or not. |

The tab, the selected job, and the filter live in the URL hash, so a reload
lands where you left off and a job view can be pasted to someone else.

## Configuration

Still `~/.config/jobdash/config.toml`. `interval` sets the SSH cadence, and the
browser follows it over SSE rather than polling on its own clock.

Set `notify_url` under `[global]` to send new alerts somewhere other than the
Mac's notification centre. A Slack or Discord webhook URL is posted as JSON;
anything else — an ntfy topic, say — gets plain text. The POST runs on its own
thread, so an unreachable endpoint cannot stretch the refresh.

```toml
[global]
notify = true
notify_url = "https://ntfy.sh/your-topic"
```

## Stopping it

Once loaded, `KeepAlive` means launchd restarts the service the moment it exits,
so `pkill` will not stop it.

```bash
launchctl bootout gui/$UID/com.jmcmillen.jobdash
```

## Tests

The aggregations in `queries.py` are the easiest thing here to get subtly wrong,
so they are pinned against a seeded in-memory database.

```bash
.venv/bin/python -m pytest tests -q
```

## Retention

Raw samples land at one row per job per poll. Anything older than three days is
folded into `samples_hourly` — one row per job per hour, carrying the same
capped-gap `dt_s` the raw integration uses, so GPU-hours match across the
boundary. `prune` drops both tables past 90 days. The poller runs rollup and
prune once a day; startup no longer owns the only chance to do housekeeping.

## Known gaps

- **iruka2 reports no GPU utilization.** Both probes are ruled out. `srun
  --overlap` still times out waiting for the job step, and the compute nodes
  refuse a direct ssh (`Permission denied (publickey,password)`), so the
  node-side `nvidia-smi` fallback cannot run either. `gpu = false` stays set for
  it in `config.toml`, and the utilization chart, GPU-hours, and efficiency
  numbers remain gaivi only. `scontrol show job -d` does report the allocated
  GPU index (`GRES=gpu:...(IDX:3)`), so if node access is ever granted, that is
  the way in.
