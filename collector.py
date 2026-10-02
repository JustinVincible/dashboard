#!/usr/bin/env python3
"""Multi-cluster live Slurm dashboard.

What it does per refresh, for every configured cluster IN PARALLEL, in exactly
ONE ssh round-trip each:
  * squeue table (running / pending / completing)
  * per-running-job GPU mem + util via `srun --overlap ... nvidia-smi`
  * resolves Slurm's StdOut launcher stub to the real `<stem>_<run>.log`
  * tails that log's last line AND greps its FULL history for metric values,
    so loss sparklines are correct on the very first draw (no warm-up)
  * log mtime + an error scan (OOM / CUDA / Traceback / NaN) over the tail
  * `sacct` panel of recently finished jobs with exit codes and efficiency

Then it raises alerts: dead/stalled logs, NaN or exploding loss, OOM and CUDA
errors, jobs nearly out of walltime, and chronically idle GPUs. Optionally
fires a macOS notification the first time each alert appears.

Usage:
  jobdash.py                       # live loop, every 20s
  jobdash.py --once                # single draw (scriptable / pipeable)
  jobdash.py --cluster NAME        # restrict to one cluster (repeatable)
  jobdash.py --history 3           # finished-jobs panel, last 3 days
  jobdash.py --logs                # resolved log paths only (old jobs.py mode)
  jobdash.py --fetch ./logs        # scp every resolved log locally
  jobdash.py --write-config        # drop a starter config and exit

Config: ~/.config/jobdash/config.toml  (see --write-config)
State:  ~/.local/state/jobdash/history.jsonl  (append-only metric snapshots)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

# ---------------------------------------------------------------- config ----

CONFIG_PATH = Path(os.environ.get("JOBDASH_CONFIG", Path.home() / ".config/jobdash/config.toml"))
STATE_PATH = Path.home() / ".local/state/jobdash/history.jsonl"

DEFAULT_CONFIG = """\
# jobdash configuration
[global]
interval = 20            # seconds between refreshes
history_days = 2         # sacct look-back for the finished-jobs panel
stall_minutes = 20       # log untouched this long => STALLED alert
low_util_pct = 20        # GPU util below this counts as idle
timeleft_warn_min = 30   # warn when a job has less walltime than this
notify = false           # macOS notification on new alerts (osascript)
# notify_url = "https://ntfy.sh/your-topic"   # also POST new alerts here.
# A Slack or Discord webhook URL is sent as JSON; anything else as plain text.
metric_points = 120      # max metric samples pulled per job for the sparkline
price_per_kwh = 0.12     # electricity rate, for the energy panel

# One [[cluster]] block per machine. Edit these before the first run.
# `host` is an ssh alias from ~/.ssh/config; `user` is your account there.
[[cluster]]
name = "mycluster"
host = "mycluster"
user = "myusername"
# slots = 6              # how many jobs you may run at once here; drives the
                         # "compute in use" panel. Omit to leave this host out.
# Regexes with ONE capture group, matched against the whole log.
# The first key is the one that gets the sparkline.
metrics = { loss = "loss[=: ]+([0-9]*[.]?[0-9]+([eE][+-]?[0-9]+)?)" }

# A second cluster where the GPU probe is blocked or slow.
# [[cluster]]
# name = "othercluster"
# host = "othercluster"
# user = "myusername"
# gpu = false
# node_ssh = false       # skip the ssh-to-compute-node GPU fallback

# A workstation with no scheduler. Every tmux pane running something counts as
# a job; see tmuxhost.py for how the columns map.
# [[cluster]]
# name = "workstation"
# host = "workstation"
# user = "myusername"
# kind = "tmux"
# slots = 2              # on a tmux host a slot is a GPU, not a pane
"""


@dataclass
class Cluster:
    name: str
    host: str
    user: str
    kind: str = "slurm"           # "slurm" | "tmux" (a workstation, see tmuxhost.py)
    metrics: dict = field(default_factory=lambda: {"loss": r"loss[=: ]+([0-9]*[.]?[0-9]+([eE][+-]?[0-9]+)?)"})
    connect_timeout: int = 15     # ssh ConnectTimeout
    command_timeout: int = 60     # wall-clock budget for the whole remote script
    batch_mode: bool = True       # False => allow ssh to prompt (breaks the loop, use for debugging)
    multiplex: bool = False       # ControlMaster reuse; off by default, see ssh_opts_for
    gpu: bool = True              # per-cluster: skip srun --overlap if it is slow/blocked here
    gpu_timeout: int = 12         # hard cap on the GPU probe; it can never hang a refresh
    node_ssh: bool = True         # fall back to ssh-ing the compute node when srun fails
    slots: int = 0                # how much of this host you may occupy at once; 0 = untracked
    slot_by: str = ""             # what a slot counts: "job" or "gpu". "" => by kind, see slot_unit
    ssh_opts: list = field(default_factory=list)   # extra raw ssh flags


def slot_unit(c: Cluster) -> str:
    """What one slot means on this host. A scheduler rations jobs, a workstation GPUs."""
    return c.slot_by or ("gpu" if c.kind == "tmux" else "job")


@dataclass
class Settings:
    interval: int = 20
    history_days: int = 2
    stall_minutes: int = 20
    low_util_pct: int = 20
    timeleft_warn_min: int = 30
    notify: bool = False
    notify_url: str = ""          # ntfy topic or Slack/Discord webhook, "" to skip
    metric_points: int = 120
    price_per_kwh: float = 0.12   # what a kWh costs, for the energy panel
    clusters: list = field(default_factory=list)


def load_config() -> Settings:
    if not CONFIG_PATH.exists():
        sys.exit(f"no config at {CONFIG_PATH} — run `collector.py --write-config` "
                 f"and edit the [[cluster]] blocks in it")
    if tomllib is None:
        sys.exit("python 3.11+ required to read the TOML config")
    data = tomllib.loads(CONFIG_PATH.read_text())
    g = data.get("global", {})
    s = Settings(**{k: v for k, v in g.items() if k in Settings.__dataclass_fields__})
    for c in data.get("cluster", []):
        s.clusters.append(Cluster(
            name=c.get("name") or c["host"],
            host=c["host"],
            user=c["user"],
            metrics=c.get("metrics") or Cluster.metrics.default_factory(),  # type: ignore[attr-defined]
            **{k: v for k, v in c.items()
               if k in ("kind", "connect_timeout", "command_timeout", "batch_mode",
                        "multiplex", "gpu", "gpu_timeout", "node_ssh", "ssh_opts",
                        "slots", "slot_by")},
        ))
    if not s.clusters:
        sys.exit(f"no [[cluster]] blocks in {CONFIG_PATH}")
    return s


# ------------------------------------------------------------- ssh layer ----

# Login banners some clusters print on every connection; never real output.
NOISE = re.compile(r"post-quantum|store now|openssh\.com/pq|^\*\*|Authorized users only|^Last login")
FS = "\x1f"   # field sep: cannot appear inside a log line
RS = "\x1e"   # record sep for multi-value fields


DEBUG = False


@dataclass
class SSHResult:
    ok: bool
    out: str = ""
    err: str = ""
    rc: int = 0
    reason: str = ""      # human-readable failure cause, empty when ok


def diagnose(rc: int, err: str, timed_out: bool, timeout: int) -> str:
    """Turn an ssh failure into something actionable instead of 'unreachable'."""
    e = err.lower()
    if timed_out:
        return (f"command timed out after {timeout}s (connection is fine if `ssh` works "
                f"interactively — more likely a hung srun/stat on a slow FS; try --no-gpu)")
    if "permission denied" in e or "publickey" in e:
        return ("permission denied — BatchMode blocks password/passphrase prompts. "
                "Add the key to ssh-agent (`ssh-add`), or set batch_mode = false for this cluster")
    if "could not resolve" in e or "name or service not known" in e:
        return "hostname did not resolve (ssh alias missing from the config this process sees?)"
    if "connection timed out" in e or "operation timed out" in e:
        return "TCP connect timed out — VPN down, or raise connect_timeout"
    if "connection refused" in e:
        return "connection refused"
    if "control socket" in e or "multiplex" in e or "controlpath" in e:
        return "ssh connection-multiplexing conflict — set multiplex = false for this cluster"
    if "host key verification" in e:
        return "host key verification failed"
    if rc == 127:
        return "remote command not found (squeue/sacct not on the non-interactive PATH?)"
    first = next((l for l in err.splitlines() if l.strip()), "")
    return f"ssh exited {rc}: {first[:160]}" if first else f"ssh exited {rc} with no output"


def ssh(host: str, cmd: str, timeout: int = 30, opts=None) -> SSHResult:
    """Run one remote command. Always reports WHY it failed."""
    argv = ["ssh"] + list(opts or ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]) + [host, cmd]
    if DEBUG:
        print(f"[debug] {' '.join(argv[:-1])} <script {len(cmd)}B>", file=sys.stderr)
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return SSHResult(False, reason=diagnose(0, "", True, timeout))
    except FileNotFoundError:
        return SSHResult(False, reason="ssh binary not found in PATH")
    out = "\n".join(l for l in r.stdout.splitlines() if not NOISE.search(l))
    err = "\n".join(l for l in r.stderr.splitlines() if not NOISE.search(l))
    if DEBUG and err:
        print(f"[debug] {host} stderr: {err[:800]}", file=sys.stderr)
    if r.returncode and not out.strip():
        return SSHResult(False, out, err, r.returncode, diagnose(r.returncode, err, False, timeout))
    return SSHResult(True, out, err, r.returncode)


def ssh_opts_for(cl) -> list:
    """Per-cluster ssh flags. Multiplexing is OFF by default: it is a nice
    speedup but it silently breaks setups that already define their own
    ControlPath, or whose socket path is too long for a unix socket.
    """
    o = ["-o", f"ConnectTimeout={cl.connect_timeout}"]
    if cl.batch_mode:
        o += ["-o", "BatchMode=yes"]
    if cl.multiplex:
        o += ["-o", "ControlMaster=auto",
              "-o", f"ControlPath={Path.home()}/.ssh/cm-%r@%h:%p",
              "-o", "ControlPersist=300"]
    return o + list(cl.ssh_opts)


# ------------------------------------------------------------ collection ----

SQ_FMT = "%.18i|%.14P|%.34j|%.2t|%.10M|%.10L|%A|%R"


def build_remote_script(user: str, metric_re: str, points: int, history_days: int,
                        want_gpu: bool, want_logs: bool, gpu_timeout: int = 12,
                        want_cluster: bool = True, want_mem: bool = True,
                        node_ssh: bool = True) -> str:
    """One script, one round-trip: squeue, then per-running-job detail in
    parallel subshells, then sacct history. A serial per-job ssh costs ~1-2s of
    channel setup each; batching keeps a refresh flat regardless of job count.

    Everything that can block is wrapped in `timeout` and its stderr is captured
    into a field rather than allowed to escape: a job step that will not start
    must degrade to "gpu: n/a (reason)" on ONE ROW, never take the cluster down.
    """
    # scontrol is needed for the log stem AND for the node fallback, so it runs
    # whenever either half is wanted.
    info_block = ''
    if want_gpu or want_logs:
        info_block = (
            '  info=$(TO_ scontrol show job $real 2>/dev/null | tr " " "\\n")\n'
            '  node=$(printf "%s\\n" "$info" | sed -n "s/^BatchHost=//p" | head -1)\n'
        )

    gpu_block = ''
    if want_gpu:
        # power.draw is what the board is actually pulling; name keys the TDP
        # fallback for cards that report it as [N/A].
        q = ('nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,'
             'power.draw,power.limit,name --format=csv,noheader,nounits')
        gpu_block = (
            # Attempt 1: a job step inside the allocation. Captures stderr so a
            # refusal ("Requested nodes are busy", "job step creation temporarily
            # disabled") is reported instead of lost.
            '  gpu=$(TO_ srun --overlap --quiet --jobid=$real ' + q + ' 2>&1 | head -8 | tr "\\n" ";")\n'
        )
    if want_gpu and node_ssh:
        gpu_block += (
            # Attempt 2: many clusters let you ssh a node you hold an allocation
            # on, and it costs nothing when attempt 1 already worked. It runs
            # outside the job step, so it sees every card on the node -- the
            # scope marker keeps that distinguishable downstream.
            '  case "$gpu" in\n'
            '    *[0-9]*,*[0-9]*,*[0-9]*) ;;\n'
            '    *) if [ -n "$node" ]; then\n'
            '         gpu2=$(TO_ ssh -o BatchMode=yes -o ConnectTimeout=5 '
            '-o StrictHostKeyChecking=no "$node" ' + shlex.quote(q) + ' 2>&1 | head -8 | tr "\\n" ";")\n'
            '         case "$gpu2" in *[0-9]*,*[0-9]*,*[0-9]*) gpu="$gpu2"; scope=node ;; esac\n'
            '       fi ;;\n'
            '  esac\n'
        )

    log_block = ''
    if want_logs:
        log_block = (
            '  stdout=$(printf "%s\\n" "$info" | sed -n "s/^StdOut=//p" | head -1)\n'
            '  stem="${stdout%.*}"\n'
            '  latest=$(ls -1tr "${stem}"_*.log 2>/dev/null | tail -1)\n'
            '  [ -z "$latest" ] && latest="$stdout"\n'
            '  if [ -f "$latest" ]; then\n'
            '    lastline=$(tail -n1 "$latest" 2>/dev/null | tr -d "\\r")\n'
            '    mtime=$(stat -c %Y "$latest" 2>/dev/null || stat -f %m "$latest" 2>/dev/null)\n'
            '    errs=$(tail -n 400 "$latest" 2>/dev/null | grep -cE '
            '"out of memory|Out of memory|CUDA error|CUDA out|Traceback|RuntimeError|[^a-z]nan|[^a-z]-?inf[^a-z]")\n'
            '    series=$(grep -oiE ' + shlex.quote(metric_re) + ' "$latest" 2>/dev/null '
            '| grep -oE "[0-9]*[.]?[0-9]+([eE][+-]?[0-9]+)?" | tail -' + str(points) + ' | tr "\\n" ";")\n'
            '  fi\n'
        )

    # Cluster-wide state, aggregated remotely. Shipping raw squeue for a busy
    # cluster every 20s would undo the round-trip budget this script exists for.
    cluster_block = ''
    if want_cluster:
        cluster_block = (
            'echo "@@QUEUE"\n'
            "squeue -h -o '%T|%P' 2>/dev/null | sort | uniq -c\n"
            'echo "@@SINFO"\n'
            "sinfo -h -O 'Partition:24,StateLong:16,Nodes:8,CPUsState:20,"
            "Gres:40,GresUsed:40' 2>/dev/null\n"
            # A node belongs to several partitions, so the view above counts it
            # once per partition. The per-node view, deduped on node name, is
            # what the cluster totals are built from.
            'echo "@@NODES"\n'
            "sinfo -h -N -O 'NodeList:24,StateLong:16,CPUsState:20,Gres:40,"
            "GresUsed:40' 2>/dev/null | awk '!seen[$1]++'\n"
        )

    # MaxRSS lives on the .batch step and ReqMem on the allocation line, so this
    # cannot ride on the -X history query below. The grep drops the srun steps.
    mem_block = ''
    if want_mem:
        keep = shlex.quote(r'^[0-9][0-9_]*(\.batch)?\|')
        mem_block = (
            'echo "@@MEM"\n'
            'sacct -u ' + user + ' -S now-' + str(history_days) + 'days -n -P '
            '-o JobID,ReqMem,MaxRSS 2>/dev/null | grep -E ' + keep + ' | tail -600\n'
        )

    body = (
        # `timeout` is coreutils and not universal; degrade to running bare
        # rather than failing every command with "TO_: not found".
        'if command -v timeout >/dev/null 2>&1; then TO_() { timeout ' + str(gpu_timeout) + ' "$@"; };'
        ' else TO_() { "$@"; }; fi\n'
        'echo "@@SQUEUE"\n'
        'squeue -u ' + user + " -h -o '" + SQ_FMT + "'\n"
        'echo "@@DETAIL"\n'
        'for real in $(squeue -u ' + user + " -h -o '%A' -t R); do\n"
        '(\n'
        '  gpu=""; node=""; latest=""; lastline=""; mtime=0; errs=""; series=""; scope=job\n'
        + info_block + gpu_block + log_block +
        '  printf "%s' + FS + '%s' + FS + '%s' + FS + '%s' + FS + '%s' + FS + '%s' + FS + '%s' + FS + '%s' + FS + '%s\\n" '
        '"$real" "$gpu" "$(basename "$latest" 2>/dev/null)" "$mtime" "$errs" "$series" "$node" "$lastline" "$scope"\n'
        ') &\n'
        'done\n'
        'wait\n'
        + cluster_block +
        'echo "@@SACCT"\n'
        # Submit and Start make queue waits exact. A job that pends and starts
        # between two polls is invisible to the observed transition, which is
        # why array jobs used to leave the wait panel empty.
        'sacct -u ' + user + ' -S now-' + str(history_days) + 'days -X -n -P '
        '-o JobID,JobName%40,State,Elapsed,ExitCode,End,MaxRSS,Submit,Start '
        '2>/dev/null | tail -300\n'
        + mem_block +
        'echo "@@NOW"\n'
        'date +%s\n'
    )
    return body


JID_OK = re.compile(r"^[0-9][0-9_\[\]\-]*$")


def log_tail(cl: Cluster, jid: str, lines: int = 80) -> tuple[str, str]:
    """Last N lines of a running job's log, from whichever backend owns it."""
    if cl.kind == "tmux":
        import tmuxhost
        return tmuxhost.log_tail(cl, jid, lines)
    return log_tail_slurm(cl, jid, lines)


def log_tail_slurm(cl: Cluster, jid: str, lines: int = 80) -> tuple[str, str]:
    """Fetch the last N lines of a running job's log. -> (path, text).

    On demand only. Shipping this many lines for every job on every tick would
    multiply the refresh payload for output nobody is reading.

    The job id lands in a remote shell command, so it is matched against a
    numeric-and-brackets pattern first and quoted after. Nothing else from the
    request reaches the remote side.
    """
    if not JID_OK.match(jid):
        return "", "(refusing a job id that is not numeric)"
    lines = max(1, min(500, int(lines)))
    q = shlex.quote(jid)
    script = (
        'info=$(scontrol show job ' + q + ' 2>/dev/null | tr " " "\n")\n'
        'stdout=$(printf "%s\\n" "$info" | sed -n "s/^StdOut=//p" | head -1)\n'
        'stem="${stdout%.*}"\n'
        'latest=$(ls -1tr "${stem}"_*.log 2>/dev/null | tail -1)\n'
        '[ -z "$latest" ] && latest="$stdout"\n'
        'echo "@@FILE $latest"\n'
        '[ -f "$latest" ] && tail -n ' + str(lines) + ' "$latest"\n'
    )
    res = ssh(cl.host, script, timeout=cl.command_timeout, opts=ssh_opts_for(cl))
    if not res.ok:
        return "", f"({res.reason})"
    out = res.out.splitlines()
    path = ""
    for i, line in enumerate(out):
        if line.startswith("@@FILE "):
            path = line[7:].strip()
            out = out[i + 1:]
            break
    return path, "\n".join(out)


SQUEUE_FMT = "%.16i %.14P %.32j %.20u %.2t %.11M %.12b %R"


def squeue_all(cl: Cluster) -> str:
    """The whole cluster's queue, every user, running jobs first."""
    cmd = f"squeue -S -t,u -o {shlex.quote(SQUEUE_FMT)}"
    res = ssh(cl.host, cmd, timeout=cl.command_timeout, opts=ssh_opts_for(cl))
    return res.out if res.ok else f"({res.reason})"


# Board TDP in watts, keyed by a substring of the nvidia-smi model name. Used
# only when the card will not report power.draw; matched longest-first so
# "A100-SXM4-80GB" does not fall through to a shorter "A100" entry.
TDP_W = {
    "H100": 700, "A100-SXM4-80GB": 400, "A100-SXM4": 400, "A100-PCIE": 250,
    "A100": 300, "A40": 300, "A6000": 300, "RTXPRO6000": 600, "6000ADA": 300,
    "TITANRTX": 280, "TITAN RTX": 280, "2080TI": 250, "1080TI": 250,
    "V100": 300, "L40S": 350, "L40": 300, "T4": 70,
}


def tdp_for(name: str) -> int | None:
    """Board TDP for a model name, or None when the model is unknown."""
    n = (name or "").upper().replace("NVIDIA ", "").replace("-", "")
    for key in sorted(TDP_W, key=len, reverse=True):
        if key.replace("-", "").replace(" ", "") in n.replace(" ", ""):
            return TDP_W[key]
    return None


def _num(x: str) -> float | None:
    """nvidia-smi writes an unsupported field as [N/A], so fields are checked
    one at a time rather than all-or-nothing across the row."""
    x = x.strip()
    try:
        return float(x)
    except ValueError:
        return None


def parse_gpu(raw: str):
    """`nvidia-smi` csv, possibly several GPUs, possibly an error message.

    Returns (stats, error_text). stats is
    (used_mib, total_mib, mean_util, n_gpus, watts, est_watts, name), summed
    across the GPUs the job holds; error_text is whatever non-numeric complaint
    came back instead.

    Rows are 6 fields now (memory, util, power.draw, power.limit, name) and 3 in
    older captures, so both widths parse. `watts` is measured and stays None
    unless every card reported it; `est_watts` is TDP x util and is kept
    separate so an estimate is never summed into a measurement.
    """
    used = total = 0
    utils, watts, est = [], [], []
    name = ""
    for chunk in raw.split(";"):
        f = [x.strip() for x in chunk.split(",")]
        if len(f) not in (3, 6):
            continue
        u, t, ut = _num(f[0]), _num(f[1]), _num(f[2])
        if u is None or t is None or ut is None:
            continue
        used += int(u)
        total += int(t)
        utils.append(ut)
        if len(f) == 6:
            name = name or f[5]
            w = _num(f[3])
            if w is not None:
                watts.append(w)
            tdp = tdp_for(f[5]) or _num(f[4])
            if tdp:
                est.append(tdp * ut / 100.0)

    if utils:
        # A partial power reading would understate the total, so it is dropped.
        measured = round(sum(watts), 1) if len(watts) == len(utils) else None
        estimated = round(sum(est), 1) if len(est) == len(utils) else None
        return (used, total, int(sum(utils) / len(utils)), len(utils),
                measured, estimated, name), ""
    msg = " ".join(raw.replace(";", " ").split())
    if not msg:
        return None, ""
    msg = re.sub(r"^srun:\s*", "", msg)
    return None, msg[:70]


GRES_RE = re.compile(r"gpu:([A-Za-z0-9_.\-]+):(\d+)")


def _gres_counts(field: str) -> dict:
    """`gpu:A100:3` / `gpu:A100:2(IDX:0,2)` -> {"A100": 3}. A node with no GPUs
    reports "(null)"."""
    out: dict = {}
    for model, n in GRES_RE.findall(field or ""):
        out[model] = out.get(model, 0) + int(n)
    return out


IDLE_STATES = ("idle",)
BUSY_STATES = ("allocated", "mixed", "completing")


def parse_cluster(queue_lines: list, sinfo_lines: list, node_lines: list) -> dict:
    """Cluster-wide queue and node state -> {partition: row}.

    Split out of collect() so it is testable against captured output with no
    ssh, the way tmuxhost.parse() already is.

    sinfo emits one row per partition+state pair, so partitions repeat and are
    folded into one row here. The default partition carries a trailing "*".
    A pending job may name several partitions ("Quick,Contributors"); it counts
    toward each.

    The synthetic "ALL" row is built from the deduped per-node view, not by
    summing partitions -- a node in three partitions would be counted three
    times.
    """
    parts: dict = {}

    def row(name):
        return parts.setdefault(name, {
            "partition": name, "running": 0, "pending": 0, "other": 0,
            "nodes_total": 0, "nodes_alloc": 0, "nodes_idle": 0,
            "nodes_mixed": 0, "nodes_down": 0,
            "cpus_alloc": 0, "cpus_idle": 0, "cpus_total": 0,
            "gpus_total": 0, "gpus_alloc": 0, "gpu_models": {},
        })

    total = row("ALL")
    for line in queue_lines:
        f = line.split()
        if len(f) != 2 or not f[0].isdigit():
            continue
        n, rest = int(f[0]), f[1]
        state, _, plist = rest.partition("|")
        key = {"RUNNING": "running", "PENDING": "pending"}.get(state.strip(), "other")
        total[key] += n
        for part in plist.split(","):
            part = part.strip().rstrip("*")
            if part:
                row(part)[key] += n

    def add_nodes(r, state, n, cpus, gres, used):
        r["nodes_total"] += n
        if state.startswith("idle"):
            r["nodes_idle"] += n
        elif state.startswith("mix"):
            r["nodes_mixed"] += n
        elif state.startswith(("alloc", "completing")):
            r["nodes_alloc"] += n
        else:                       # down, drained, draining, reserved, ...
            r["nodes_down"] += n
        a, i, _o, t = (cpus.split("/") + ["0"] * 4)[:4]
        r["cpus_alloc"] += int(a) * n if a.isdigit() else 0
        r["cpus_idle"] += int(i) * n if i.isdigit() else 0
        r["cpus_total"] += int(t) * n if t.isdigit() else 0
        for model, c in _gres_counts(gres).items():
            r["gpus_total"] += c * n
            r["gpu_models"][model] = r["gpu_models"].get(model, 0) + c * n
        r["gpus_alloc"] += sum(_gres_counts(used).values()) * n

    # Partition rows: Partition StateLong Nodes CPUsState Gres GresUsed
    for line in sinfo_lines:
        f = line.split()
        if len(f) < 4 or not f[2].isdigit():
            continue
        f += [""] * (6 - len(f))
        add_nodes(row(f[0].rstrip("*")), f[1].lower(), int(f[2]), f[3], f[4], f[5])

    # Cluster totals: NodeList StateLong CPUsState Gres GresUsed, one per node.
    for line in node_lines:
        f = line.split()
        if len(f) < 3:
            continue
        f += [""] * (5 - len(f))
        add_nodes(total, f[1].lower(), 1, f[2], f[3], f[4])

    return parts


def parse_mem(lines: list) -> dict:
    """`sacct -o JobID,ReqMem,MaxRSS` -> {jid: (req_mem, max_rss)}.

    ReqMem sits on the allocation line and MaxRSS on the .batch step, so the two
    halves are joined on the base job id. Either half may be missing while the
    job is still running.
    """
    out: dict = {}
    for line in lines:
        f = line.split("|")
        if len(f) < 3:
            continue
        jid, req, rss = f[0].strip(), f[1].strip(), f[2].strip()
        base = jid.split(".")[0]
        if not base:
            continue
        cur = out.setdefault(base, ["", ""])
        if req:
            cur[0] = req
        if rss:
            cur[1] = rss
    return {k: tuple(v) for k, v in out.items() if v[0] or v[1]}


@dataclass
class Job:
    cluster: str
    jid: str
    partition: str
    name: str
    state: str
    elapsed: str
    timeleft: str
    real: str
    reason: str
    # (used_mib, total_mib, mean_util_pct, n_gpus, watts, est_watts, gpu_name)
    gpu: tuple | None = None
    gpu_err: str = ""              # why the GPU query came back empty
    gpu_scope: str = "job"         # "node" when the reading covers the whole box
    node: str = ""
    log: str | None = None
    mtime: int = 0
    errors: int = 0
    series: list = field(default_factory=list)
    lastline: str = ""


@dataclass
class Snapshot:
    cluster: str
    ok: bool
    jobs: list = field(default_factory=list)
    finished: list = field(default_factory=list)
    remote_now: int = 0
    error: str = ""
    warn: str = ""
    cpu_watts: float | None = None                      # host CPU package draw (RAPL)
    cluster_stats: dict = field(default_factory=dict)   # partition -> counts
    mem: dict = field(default_factory=dict)             # jid -> (req_mem, max_rss)


SACCT_FIELDS = 9   # JobID,JobName,State,Elapsed,ExitCode,End,MaxRSS,Submit,Start


def split_sacct(line: str) -> list[str]:
    """Split one `sacct -P` row.

    JobName is user text and "|" is the separator, so a job named "a|b" would
    shift every field after it. Only the name can hold a separator, so extras
    fold back into it.
    """
    f = line.split("|")
    if len(f) > SACCT_FIELDS:
        tail = SACCT_FIELDS - 2
        f = [f[0], "|".join(f[1:len(f) - tail])] + f[len(f) - tail:]
    return f


def collect(cl: Cluster, st: Settings, want_gpu: bool, want_logs: bool,
            history_days: int) -> Snapshot:
    """One refresh of one machine. `kind` picks the backend; the Snapshot it
    returns is the same shape either way, so nothing downstream branches."""
    if cl.kind == "tmux":
        import tmuxhost
        return tmuxhost.collect(cl, st, want_gpu, want_logs)
    return collect_slurm(cl, st, want_gpu, want_logs, history_days)


def collect_slurm(cl: Cluster, st: Settings, want_gpu: bool, want_logs: bool,
                  history_days: int) -> Snapshot:
    want_gpu = want_gpu and cl.gpu
    metric_re = next(iter(cl.metrics.values()))
    script = build_remote_script(cl.user, metric_re, st.metric_points, history_days,
                                 want_gpu, want_logs, cl.gpu_timeout,
                                 node_ssh=cl.node_ssh)
    res = ssh(cl.host, script, timeout=cl.command_timeout, opts=ssh_opts_for(cl))
    if not res.ok:
        return Snapshot(cl.name, False, error=res.reason)
    out = res.out

    sections = {k: [] for k in ("SQUEUE", "DETAIL", "QUEUE", "SINFO", "NODES",
                                "SACCT", "MEM", "NOW")}
    cur = None
    for line in out.splitlines():
        if line.startswith("@@"):
            cur = line[2:]
            continue
        if cur in sections:
            sections[cur].append(line)

    jobs = []
    for row in sections["SQUEUE"]:
        f = [x.strip() for x in row.split("|")]
        if len(f) != 8:
            continue
        jobs.append(Job(cl.name, *f))

    by_real = {j.real: j for j in jobs}
    for line in sections["DETAIL"]:
        f = line.split(FS)
        if len(f) != 9:
            continue
        real, gpu_raw, logname, mtime, errs, series, node, lastline, scope = f
        j = by_real.get(real.strip())
        if not j:
            continue
        j.gpu, j.gpu_err = parse_gpu(gpu_raw)
        j.gpu_scope = scope.strip() or "job"
        j.node = node.strip()
        j.log = logname or None
        j.mtime = int(mtime) if mtime.strip().isdigit() else 0
        j.errors = int(errs) if errs.strip().isdigit() else 0
        j.series = [float(v) for v in series.split(";") if v.strip()][-st.metric_points:]
        j.lastline = lastline

    finished = []
    for line in sections["SACCT"]:
        f = split_sacct(line)
        if len(f) >= 7:
            finished.append(f)

    now = int(sections["NOW"][0]) if sections["NOW"] and sections["NOW"][0].strip().isdigit() else int(time.time())
    warn = ""
    if not jobs and res.err:
        # Connected fine but the remote script complained — do not silently
        # render this as "nothing queued".
        warn = res.err.splitlines()[0][:120]
    return Snapshot(cl.name, True, jobs, finished, now, warn=warn,
                    cluster_stats=parse_cluster(sections["QUEUE"], sections["SINFO"],
                                                sections["NODES"]),
                    mem=parse_mem(sections["MEM"]))


# ---------------------------------------------------------------- alerts ----

def minutes_left(timeleft: str) -> float | None:
    """Parse Slurm D-HH:MM:SS / HH:MM:SS / MM:SS into minutes."""
    t = timeleft.strip()
    if not t or t in ("UNLIMITED", "INVALID", "NOT_SET"):
        return None
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    bits = [int(x) for x in t.split(":") if x.isdigit()]
    if len(bits) == 3:
        h, m, s = bits
    elif len(bits) == 2:
        h, (m, s) = 0, bits
    else:
        return None
    return days * 1440 + h * 60 + m + s / 60


def alerts_for(snap: Snapshot, st: Settings) -> list:
    """Shared with the web service. Deferred import — alerts.py imports this."""
    from alerts import alerts_for as _rules
    return [(kind, msg) for kind, msg, _c, _j in _rules(snap, st)]


def notify(alerts, seen: set):
    fresh = [a for a in alerts if a[1] not in seen]
    if not fresh or sys.platform != "darwin":
        seen.update(a[1] for a in alerts)
        return
    body = "; ".join(a[1] for a in fresh[:3]).replace('"', "'")
    subprocess.run(["osascript", "-e",
                    f'display notification "{body}" with title "jobdash"'],
                   capture_output=True)
    seen.update(a[1] for a in alerts)


def record(snaps, path: Path = STATE_PATH):
    """Append-only metric snapshots so a run's history survives restarts."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for s in snaps:
                for j in s.jobs:
                    if j.series:
                        fh.write(json.dumps({"t": int(time.time()), "cluster": s.cluster,
                                             "jid": j.jid, "name": j.name.strip(),
                                             "last": j.series[-1], "n": len(j.series)}) + "\n")
    except OSError:
        pass


# -------------------------------------------------------------- rendering ---

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN, MAG = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[35m"
SPARK = "▁▂▃▄▅▆▇█"
SEV = {"NAN": RED, "ERR": RED, "DIVERGE": RED, "STALL": YELLOW, "WALL": YELLOW, "IDLE": DIM}


def sparkline(vals, width=40):
    if len(vals) < 2:
        return ""
    v = vals[-width:]
    lo, hi = min(v), max(v)
    if hi == lo:
        return SPARK[0] * len(v)
    return "".join(SPARK[min(7, int((x - lo) / (hi - lo) * 7))] for x in v)


def state_color(s):
    return {"R": GREEN, "PD": YELLOW, "CG": CYAN}.get(s, RED)


def util_color(p):
    return DIM if p is None else GREEN if p >= 60 else YELLOW if p >= 20 else RED


STATE_ORDER = {"R": 0, "CG": 1, "PD": 2}


def render(snaps, st: Settings, show_history: int, width: int, split: bool = False) -> str:
    """One merged table across every cluster by default (--split restores the
    per-cluster sections). Sorted running-first, then by cluster, then by id.
    """
    L = [f"{BOLD}{CYAN}Slurm dashboard{RESET}  {DIM}{time.strftime('%Y-%m-%d %H:%M:%S')}{RESET}"]

    down = [s for s in snaps if not s.ok]
    up = [s for s in snaps if s.ok]
    cw = max([len(s.cluster) for s in snaps] + [7])

    for s in down:
        L.append(f"  {RED}{s.cluster:<{cw}} {s.error}{RESET}")
    for s in up:
        if s.warn:
            L.append(f"  {YELLOW}{s.cluster:<{cw}} remote stderr: {s.warn}{RESET}")

    groups = ([(s.cluster, s.jobs, s) for s in up] if split
              else [(None, [j for s in up for j in s.jobs], None)])

    all_alerts = [a for s in up for a in alerts_for(s, st)]
    now_by_cluster = {s.cluster: s.remote_now for s in up}

    for label, jobs, snap in groups:
        jobs = sorted(jobs, key=lambda j: (STATE_ORDER.get(j.state, 9), j.cluster, j.jid))
        L.append("")
        if label:
            n_r = sum(1 for j in jobs if j.state == "R")
            L.append(f"{BOLD}{label}{RESET}  {DIM}{n_r} running / {len(jobs)} total{RESET}")
        else:
            tot = {c: 0 for c in (s.cluster for s in up)}
            run = dict(tot)
            for j in jobs:
                tot[j.cluster] += 1
                run[j.cluster] += j.state == "R"
            bits = "   ".join(f"{c}: {GREEN}{run[c]}{RESET}R/{tot[c]}" for c in tot)
            L.append(f"{BOLD}all clusters{RESET}  {DIM}{sum(tot.values())} jobs{RESET}   {bits}")
        L.append("─" * width)

        if not jobs:
            L.append(f"{DIM}  nothing queued or running{RESET}")
            continue

        cluster_col = "" if split else f"{'HOST':<{cw}} "
        L.append(f"{BOLD}{cluster_col}{'JOBID':<10} {'NAME':<28} {'ST':<3} "
                 f"{'ELAPSED':<9} {'LEFT':<9} {'GPU MEM':<14} {'UTIL':<5} NODE / REASON{RESET}")
        for j in jobs:
            c = state_color(j.state)
            if j.state != "R":
                mem, util, uc = "", "", RESET
            elif j.gpu:
                used, total, pct = j.gpu[:3]
                uc, mem, util = util_color(pct), f"{used}/{total}MiB", f"{pct}%"
            else:
                # A failed GPU probe is one dim cell, not a dead cluster.
                mem, util, uc = f"{DIM}n/a{RESET}", "", RESET
            pre = "" if split else f"{DIM}{j.cluster:<{cw}}{RESET} "
            L.append(f"{pre}{j.jid:<10} {j.name[:28]:<28} {c}{j.state:<3}{RESET} "
                     f"{j.elapsed:<9} {j.timeleft:<9} {mem:<14} {uc}{util:<5}{RESET} "
                     f"{j.node or j.reason}")
            if j.state == "R" and not j.gpu and j.gpu_err:
                indent = " " * (cw + 1) if not split else ""
                L.append(f"{indent}  {DIM}gpu: {j.gpu_err}{RESET}")

        live = [j for j in jobs if j.state == "R"]
        if live:
            L.append("")
            for j in live:
                tag = f"{DIM}{j.cluster}{RESET} " if not split else ""
                if not j.log:
                    L.append(f"  {tag}{j.jid} {DIM}no log yet — still initialising{RESET}")
                    continue
                rn = now_by_cluster.get(j.cluster, int(time.time()))
                age = (rn - j.mtime) / 60 if j.mtime else None
                ac = RED if age and age > st.stall_minutes else DIM
                agestr = f"{ac}({age:.0f}m ago){RESET}" if age is not None else ""
                L.append(f"  {tag}{j.jid} {DIM}{j.log}{RESET} {agestr}")
                L.append(f"    {j.lastline[: width - 6]}")
                if len(j.series) >= 2:
                    win = j.series[-40:]
                    s0, s1 = win[0], win[-1]
                    d = s1 - s0
                    dc = GREEN if d < 0 else RED if d > 0 else DIM
                    L.append(f"    loss {CYAN}{sparkline(j.series)}{RESET} {s1:.4g}"
                             f" {dc}({'+' if d >= 0 else ''}{d:.4g} over {len(win)} pts){RESET}")

    if all_alerts:
        L.append("")
        L.append(f"{BOLD}{YELLOW}⚠ alerts{RESET}")
        for kind, msg in all_alerts:
            L.append(f"  {SEV.get(kind, YELLOW)}{kind:<8}{RESET} {msg}")

    if show_history:
        L.append("")
        L.append(f"{BOLD}recently finished{RESET} {DIM}(last {show_history}d, newest last){RESET}")
        L.append("─" * width)
        rows = []
        for snap in up:
            for f in snap.finished:
                rows.append((snap.cluster, (list(f) + [""] * 7)[:7]))
        rows = rows[-14:]
        for cluster, (jid, name, state, elapsed, exitcode, end, maxrss) in rows:
            sc = (GREEN if state.startswith("COMPLETED")
                  else RED if state.startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF", "NODE_FAIL"))
                  else YELLOW)
            L.append(f"  {DIM}{cluster:<{cw}}{RESET} {jid:<12} {name[:26]:<26} "
                     f"{sc}{state[:11]:<11}{RESET} {elapsed:<10} "
                     f"{DIM}exit {exitcode:<6} {end[:16]}{RESET}")
        if not rows:
            L.append(f"{DIM}  nothing in the window{RESET}")

    return "\n".join(L)


# ------------------------------------------------------------------ modes ---

def resolve_logs(clusters, fetch: str | None):
    """Old jobs.py behaviour: print resolved log paths, optionally scp them."""
    for cl in clusters:
        print(f"\n=== {cl.name} ===")
        if cl.kind == "tmux":
            print("  n/a — tmux jobs resolve their log from fd 1, not from scontrol")
            continue
        res = ssh(cl.host, f"""
for real in $(squeue -u {cl.user} -h -o '%A' -t R); do
  stdout=$(scontrol show job $real 2>/dev/null | tr " " "\\n" | sed -n "s/^StdOut=//p")
  stem="${{stdout%.*}}"
  for lg in $(ls -1tr "${{stem}}"_*.log 2>/dev/null); do echo "$real $lg"; done
done""", timeout=cl.command_timeout, opts=ssh_opts_for(cl))
        out = res.out
        if not res.ok:
            print(f"  {res.reason}")
            continue
        if not out.strip():
            print("  no running jobs")
            continue
        print(out or "  no resolved logs")
        if fetch and out.strip():
            dest = Path(fetch) / cl.name
            dest.mkdir(parents=True, exist_ok=True)
            for line in out.splitlines():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    subprocess.run(["scp", "-q", f"{cl.host}:{parts[1]}", str(dest)], check=False)
            print(f"  fetched into {dest}")


SLURM_PROBES = [
    ("connect", "echo ok"),
    ("shell/PATH", "echo $SHELL; command -v squeue || echo 'squeue NOT on PATH'"),
    ("squeue", "squeue -u $USER -h -o '%i %t' | head -3; echo '(rc='$?')'"),
    ("sacct", "sacct -u $USER -S now-1days -X -n -P -o JobID,State 2>&1 | head -3"),
    ("scontrol", "scontrol show config 2>/dev/null | head -1"),
]

TMUX_PROBES = [
    ("connect", "echo ok"),
    ("shell/PATH", "echo $SHELL; command -v tmux || echo 'tmux NOT on PATH'"),
    ("tmux", "tmux list-sessions 2>&1 | head -5; echo '(rc='$?')'"),
    ("panes", "tmux list-panes -a -F '#{session_name} #{pane_pid} #{pane_current_command}' "
              "2>&1 | head -5"),
    ("nvidia-smi", "nvidia-smi --query-gpu=index,utilization.gpu "
                   "--format=csv,noheader,nounits 2>&1 | head -4"),
    ("proc", "test -r /proc/$$/stat && echo '/proc readable' || echo '/proc NOT readable'"),
]


def check(clusters):
    """Isolate WHERE a machine breaks: connection, then each remote command."""
    for cl in clusters:
        probes = TMUX_PROBES if cl.kind == "tmux" else SLURM_PROBES
        print(f"\n=== {cl.name} ({cl.host}, user {cl.user}, {cl.kind}) ===")
        opts = ssh_opts_for(cl)
        print(f"  ssh flags: {' '.join(opts)}")
        for label, cmd in probes:
            res = ssh(cl.host, cmd, timeout=cl.connect_timeout + 20, opts=opts)
            if not res.ok:
                print(f"  {label:<12} FAIL  {res.reason}")
                if label == "connect":
                    print("  -> stopping here; later probes would fail for the same reason")
                    break
            else:
                body = " | ".join(l.strip() for l in res.out.splitlines() if l.strip())[:150]
                print(f"  {label:<12} ok    {body}")
        # The real thing, timed.
        metric_re = next(iter(cl.metrics.values()))
        if cl.kind == "tmux":
            import tmuxhost
            script = tmuxhost.build_script(metric_re, 20, cl.gpu, True)
        else:
            script = build_remote_script(cl.user, metric_re, 20, 1, cl.gpu, True,
                                         node_ssh=cl.node_ssh)
        t0 = time.time()
        res = ssh(cl.host, script, timeout=cl.command_timeout, opts=opts)
        dt = time.time() - t0
        print(f"  full script  {'ok' if res.ok else 'FAIL'}  {dt:.1f}s"
              f"{'' if res.ok else '  ' + res.reason}")
        if res.ok and res.err:
            print(f"  remote stderr: {res.err.splitlines()[0][:160]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--cluster", action="append", help="limit to this cluster (repeatable)")
    ap.add_argument("--history", type=int, nargs="?", const=2, help="days of finished jobs to show")
    ap.add_argument("--no-logs", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--notify", action="store_true", help="macOS notification on new alerts")
    ap.add_argument("--logs", action="store_true", help="print resolved log paths and exit")
    ap.add_argument("--fetch", metavar="DEST", help="with --logs, scp them into DEST")
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--split", action="store_true", help="one table per cluster instead of merged")
    ap.add_argument("--check", action="store_true", help="diagnose each cluster step by step")
    ap.add_argument("--debug", action="store_true", help="print ssh argv and remote stderr")
    args = ap.parse_args()

    if args.write_config:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            sys.exit(f"{CONFIG_PATH} already exists — not overwriting")
        CONFIG_PATH.write_text(DEFAULT_CONFIG)
        print(f"wrote {CONFIG_PATH}")
        return

    global DEBUG
    DEBUG = args.debug

    st = load_config()
    if args.interval:
        st.interval = args.interval
    if args.notify:
        st.notify = True
    if args.cluster:
        want = set(args.cluster)
        st.clusters = [c for c in st.clusters if c.name in want or c.host in want]
        if not st.clusters:
            sys.exit("no cluster matched --cluster")

    if args.check:
        check(st.clusters)
        return

    if args.logs:
        resolve_logs(st.clusters, args.fetch)
        return

    hist_days = args.history if args.history is not None else 0
    seen_alerts: set = set()

    def frame():
        with ThreadPoolExecutor(max_workers=max(1, len(st.clusters))) as ex:
            snaps = list(ex.map(
                lambda c: collect(c, st, not args.no_gpu, not args.no_logs,
                                  hist_days or st.history_days),
                st.clusters))
        record(snaps)
        if st.notify:
            notify([a for s in snaps for a in alerts_for(s, st)], seen_alerts)
        width = shutil.get_terminal_size((110, 24)).columns
        return render(snaps, st, hist_days, width, args.split)

    if args.once:
        print(frame())
        return

    interactive = sys.stdout.isatty()
    try:
        while True:
            # Build the whole frame BEFORE touching the screen: the ssh
            # round-trip takes seconds, and clearing first leaves the terminal
            # blank for most of the interval.
            text = frame() + f"\n\n{DIM}refreshing every {st.interval}s — ctrl-c to quit{RESET}"
            sys.stdout.write("\033[H\033[2J" if interactive else "\n" + "=" * 78 + "\n")
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            time.sleep(st.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
