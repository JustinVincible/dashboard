#!/usr/bin/env python3
"""Workstation backend: tmux panes as jobs.

The workstation has no scheduler. A run is a process someone started in a tmux
pane, so this module builds the same `Snapshot` of `Job`s the Slurm path builds
and every consumer downstream — store, alerts, queries, the UI — stays unchanged.

The mapping:

| Slurm            | here                                                  |
|------------------|-------------------------------------------------------|
| job id           | `<pid>_<start epoch>`, stable per run, unique per rerun |
| squeue state     | always `R`; a pane either holds a live process or not  |
| partition        | tmux session name                                      |
| node             | `session:window.pane`, the target you would attach to  |
| walltime left    | `UNLIMITED`; nothing here gets killed on a clock        |
| StdOut           | the job's fd 1 when it is a file, else the pane buffer |
| sacct            | nothing — `store.close_vanished` writes the end instead |

One ssh round-trip per refresh, same as the Slurm path.
"""
from __future__ import annotations

import os
import re
import shlex

from collector import (FS, Cluster, Job, Settings, Snapshot, ssh,
                       ssh_opts_for, tdp_for)

# Wrappers to descend through when looking for the process that IS the job.
# A pane sitting at one of these and nothing else is idle, not a run.
SHELLS = "bash sh zsh fish tcsh csh ksh dash tmux screen su sudo login"

# Things someone is using, not running. Without this the table fills up with
# every pane holding an editor or a pager.
NOT_JOBS = ("vim vi nvim emacs nano pico micro helix less more man info htop top "
            "btop nvtop atop watch git ssh mosh sftp ranger mc gdb ipython")

ERR_RE = ("out of memory|Out of memory|CUDA error|CUDA out|Traceback|RuntimeError"
          "|[^a-z]nan|[^a-z]-?inf[^a-z]")
NUM_RE = "[0-9]*[.]?[0-9]+([eE][+-]?[0-9]+)?"

CAPTURE_LINES = 2000    # pane scrollback to scan when a job has no log file
JOB_FIELDS = 17

# tmux rewrites control characters in format output — a tab or a newline comes
# back as "_", a 0x1f as the literal text "\\037" — so the separator has to be
# printable. tmux forbids ":" in a session name but allows it in a window name,
# so the window name goes last, where `read` sweeps up whatever is left.
#
# `#{pane_pid}` is the pane's shell, never the job, so every field that
# describes the run is resolved from the process tree below it.
PANE_SEP = ":"
PANE_FMT = PANE_SEP.join([
    "#{session_name}", "#{window_index}", "#{pane_index}", "#{pane_id}",
    "#{pane_pid}", "#{window_activity}", "#{window_name}",
])


# CPU package power from the RAPL energy counter (microjoules, wraps at
# max_energy_range_uj). Two reads half a second apart give a mean draw. The
# counter is root-only on most kernels, so an unreadable file prints nothing
# and the dashboard shows no CPU figure.
CPU_BLOCK = (
    'echo "@@CPU"\n'
    'rp=/sys/class/powercap/intel-rapl:0\n'
    'if [ -r $rp/energy_uj ]; then\n'
    '  e1=$(cat $rp/energy_uj); t1=$(date +%s%N)\n'
    '  sleep 0.5\n'
    '  e2=$(cat $rp/energy_uj); t2=$(date +%s%N)\n'
    '  echo "$e1 $e2 $t1 $t2 $(cat $rp/max_energy_range_uj)"\n'
    'fi\n'
)


def build_script(metric_re: str, points: int, want_gpu: bool, want_logs: bool) -> str:
    """One script, one round-trip: GPU inventory, then a record per live pane.

    Every command that can fail on an idle box — no tmux server, no nvidia-smi,
    a pane whose process exited mid-scan — is swallowed, because `ssh()` only
    reports failure when there is no stdout at all. A quiet workstation has to
    land as "ok, zero jobs".
    """
    gpu_block = "" if not want_gpu else (
        'echo "@@GPU"\n'
        'nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,'
        'power.draw,name --format=csv,noheader,nounits 2>/dev/null || true\n'
        'echo "@@APPS"\n'
        'nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_uuid '
        '--format=csv,noheader,nounits 2>/dev/null || true\n'
    )

    log_block = "" if not want_logs else (
        # fd 1 is the whole trick. A job started as `python train.py > run.log`
        # hands us a real file, which means real mtime, a full-history metric
        # grep, and an error scan identical to the Slurm path. Only a job
        # printing straight to the terminal falls back to the pane buffer.
        '  fd1=$(readlink /proc/$jp/fd/1 2>/dev/null)\n'
        '  case "$fd1" in\n'
        '    ""|/dev/*|socket:*|pipe:*|anon_inode:*) ;;\n'
        '    *) [ -f "$fd1" ] && log="$fd1" ;;\n'
        '  esac\n'
        '  if [ -n "$log" ]; then\n'
        '    last=$(tail -n1 "$log" 2>/dev/null | tr -d "\\r" | tr "\\n" " ")\n'
        '    mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)\n'
        '    errs=$(tail -n 400 "$log" 2>/dev/null | grep -cE ' + shlex.quote(ERR_RE) + ')\n'
        '    series=$(grep -oiE ' + shlex.quote(metric_re) + ' "$log" 2>/dev/null '
        '| grep -oE ' + shlex.quote(NUM_RE) + ' | tail -' + str(points) + ' | tr "\\n" ";")\n'
        '  else\n'
        '    buf=$(tmux capture-pane -p -J -S -' + str(CAPTURE_LINES) +
        ' -t "$pane" 2>/dev/null | tr -d "\\r")\n'
        # tqdm redraws leave the visible tail blank, so the last line that says
        # anything is the one worth reporting.
        '    last=$(printf "%s\\n" "$buf" | grep -v "^[[:space:]]*$" | tail -n1 | tr "\\n" " ")\n'
        '    mtime=$act\n'
        '    errs=$(printf "%s\\n" "$buf" | tail -n 400 | grep -cE ' + shlex.quote(ERR_RE) + ')\n'
        '    series=$(printf "%s\\n" "$buf" | grep -oiE ' + shlex.quote(metric_re) +
        ' | grep -oE ' + shlex.quote(NUM_RE) + ' | tail -' + str(points) + ' | tr "\\n" ";")\n'
        '  fi\n'
    )

    return (
        'SH_=" ' + SHELLS + ' "\n'
        'SKIP_=" ' + NOT_JOBS + ' "\n'
        # The job is the shallowest non-shell process at or under the pane.
        # `tmux new-window python train.py` makes the pane process the job
        # itself; an interactive pane wraps it in a login shell, and `sudo` or
        # `env` can add more layers. Descend through those, stop at the first
        # thing that is actually running.
        'job_of() {\n'
        '  cur=$1; d=0\n'
        '  c=$(cat /proc/$cur/comm 2>/dev/null)\n'
        '  case "$SH_" in\n'
        '    *" $c "*) ;;\n'
        '    *) [ -n "$c" ] && { echo "$cur"; return 0; } ;;\n'
        '  esac\n'
        '  while [ $d -lt 8 ]; do\n'
        '    kids=$(pgrep -P "$cur" 2>/dev/null) || return 0\n'
        '    [ -z "$kids" ] && return 0\n'
        '    nxt=""\n'
        '    for k in $kids; do\n'
        '      c=$(cat /proc/$k/comm 2>/dev/null)\n'
        '      case "$SH_" in\n'
        '        *" $c "*) [ -z "$nxt" ] && nxt=$k ;;\n'
        '        *) [ -n "$c" ] && { echo "$k"; return 0; } ;;\n'
        '      esac\n'
        '    done\n'
        '    [ -z "$nxt" ] && return 0\n'
        '    cur=$nxt; d=$((d+1))\n'
        '  done\n'
        '}\n'
        # A run's GPU memory shows up under its dataloader workers and its
        # torchrun children, not under the pid we call the job.
        'desc() {\n'
        '  echo "$1"\n'
        '  for k in $(pgrep -P "$1" 2>/dev/null); do desc "$k"; done\n'
        '}\n'
        'echo "@@NOW"\n'
        'date +%s\n'
        'echo "@@BOOT"\n'
        'awk "/^btime/{print \\$2}" /proc/stat\n'
        'echo "@@HZ"\n'
        'getconf CLK_TCK\n'
        + CPU_BLOCK + gpu_block +
        'echo "@@JOBS"\n'
        'panes=$(tmux list-panes -a -F ' + shlex.quote(PANE_FMT) + ' 2>/dev/null) || true\n'
        'printf "%s\\n" "$panes" | while IFS="' + PANE_SEP +
        '" read -r s wi pi pane ppid act wn; do\n'
        '  [ -z "$ppid" ] && continue\n'
        '  jp=$(job_of "$ppid")\n'
        '  [ -z "$jp" ] && continue\n'
        '  [ -d /proc/$jp ] || continue\n'
        '  jc=$(cat /proc/$jp/comm 2>/dev/null)\n'
        '  case "$SKIP_" in *" $jc "*) continue ;; esac\n'
        '  log=""; mtime=0; errs=0; series=""; last=""\n'
        # /proc/PID/stat field 22 is start time in ticks since boot. comm can
        # hold spaces and parens, so everything up to the last ")" goes first.
        '  st=$(sed "s/.*) //" /proc/$jp/stat 2>/dev/null | awk "{print \\$20}")\n'
        '  [ -z "$st" ] && continue\n'
        '  et=$(ps -o etimes= -p $jp 2>/dev/null | tr -d " ")\n'
        '  comm=$(cat /proc/$jp/comm 2>/dev/null)\n'
        '  cmd=$(tr "\\0\\n\\r" "   " < /proc/$jp/cmdline 2>/dev/null | cut -c1-200)\n'
        '  pids=$(desc "$jp" 2>/dev/null | tr "\\n" " ")\n'
        + log_block +
        '  printf "%s' + (FS + '%s') * (JOB_FIELDS - 1) + '\\n" '
        '"$s" "$wi" "$pi" "$wn" "$pane" "$ppid" "$jp" "$st" "$et" "$comm" "$cmd" '
        '"$pids" "$log" "$mtime" "$errs" "$series" "$last"\n'
        'done\n'
    )


# ------------------------------------------------------------------ parse --

INTERPRETERS = {"python", "python3", "python2", "sh", "bash", "zsh", "env", "nohup",
                "time", "torchrun", "accelerate", "deepspeed", "uv", "poetry", "srun"}


def job_name(winname: str, comm: str, cmdline: str) -> str:
    """What to call this run in the table.

    A named tmux window is the user saying what the run is, so it wins. Failing
    that, the script being run beats the interpreter running it — every job
    would otherwise be called "python3".
    """
    w = winname.strip()
    if w and w not in SHELLS.split():
        return w
    toks = cmdline.split()
    if "-c" not in toks:
        for t in toks[1:]:
            if t.startswith("-"):
                continue
            base = os.path.basename(t)
            if base and base not in INTERPRETERS:
                return base
    return comm.strip() or "?"


def hms(seconds: float) -> str:
    """Slurm's D-HH:MM:SS. `minutes_left` parses it, which is what gives the
    Gantt strip a real start time via `store.run_start`."""
    d, r = divmod(int(seconds), 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def parse_gpu_inventory(gpu_lines: list, app_lines: list):
    """-> (uuid -> (index, util, mem_total, watts, name), pid -> [(mem_used, uuid)])

    Rows are 7 fields now and were 5 before power.draw and name were added, so
    both widths parse.
    """
    gpus, apps = {}, {}
    for line in gpu_lines:
        f = [x.strip() for x in line.split(",")]
        if len(f) in (5, 7) and f[0].isdigit():
            watts = _fnum(f[5]) if len(f) == 7 else None
            gpus[f[1]] = (int(f[0]), _num(f[4]), _num(f[3]), watts,
                          f[6] if len(f) == 7 else "")
    for line in app_lines:
        f = [x.strip() for x in line.split(",")]
        if len(f) == 3 and f[0].isdigit():
            apps.setdefault(f[0], []).append((_num(f[1]), f[2]))
    return gpus, apps


def parse_cpu_watts(lines: list) -> float | None:
    """`e1 e2 t1_ns t2_ns max` -> mean package watts, or None when unreadable."""
    try:
        e1, e2, t1, t2, wrap = (int(x) for x in lines[0].split())
    except (IndexError, ValueError):
        return None
    dt = (t2 - t1) / 1e9
    if dt <= 0:
        return None
    de = e2 - e1 if e2 >= e1 else e2 + wrap - e1
    return round(de / 1e6 / dt, 1)


def _num(x: str) -> int:
    try:
        return int(float(x))
    except ValueError:
        return 0


def _fnum(x: str) -> float | None:
    """nvidia-smi writes an unsupported field as [N/A]."""
    try:
        return float(x)
    except ValueError:
        return None


def gpu_for(pids: list, gpus: dict, apps: dict):
    """Roll a job's whole process tree up into one GPU reading.

    -> ((used_mib, total_mib, mean_util, n_gpus, watts, est_watts, name),
        "gpu0,1"), or (None, "").

    Power is per board and is attributed whole to this job, the same way
    utilization is, so two runs sharing a card each read the card's full draw.

    Memory is the job's own, summed over its processes. Utilization cannot be
    split per process — nvidia-smi reports it per device — so two runs sharing a
    card each read the card's full utilization, and `gpu_hours` will count that
    hour twice. Slurm's exclusive GRES made this impossible; a workstation does
    not, and there is nothing in the driver to divide it with.
    """
    used, uuids = 0, []
    for p in pids:
        for mem, uuid in apps.get(p, []):
            used += mem
            if uuid not in uuids:
                uuids.append(uuid)
    known = sorted(gpus[u] for u in uuids if u in gpus)
    if not known:
        return None, ""
    util = int(sum(g[1] for g in known) / len(known))
    total = sum(g[2] for g in known)
    watts = [g[3] for g in known if len(g) > 3 and g[3] is not None]
    name = next((g[4] for g in known if len(g) > 4 and g[4]), "")
    tdp = tdp_for(name)
    return ((used, total, util, len(known),
             round(sum(watts), 1) if len(watts) == len(known) else None,
             round(tdp * len(known) * util / 100.0, 1) if tdp else None,
             name),
            "gpu" + ",".join(str(g[0]) for g in known))


def collect(cl: Cluster, st: Settings, want_gpu: bool = True, want_logs: bool = True) -> Snapshot:
    want_gpu = want_gpu and cl.gpu
    metric_re = next(iter(cl.metrics.values()))
    script = build_script(metric_re, st.metric_points, want_gpu, want_logs)
    res = ssh(cl.host, script, timeout=cl.command_timeout, opts=ssh_opts_for(cl))
    if not res.ok:
        return Snapshot(cl.name, False, error=res.reason)
    return parse(cl, st, res.out, want_gpu, res.err)


def parse(cl: Cluster, st: Settings, out: str, want_gpu: bool = True,
          err: str = "") -> Snapshot:
    """Remote stdout -> Snapshot. Split out from `collect` so the field mapping
    is testable without an ssh."""
    sections: dict = {"NOW": [], "BOOT": [], "HZ": [], "CPU": [], "GPU": [], "APPS": [], "JOBS": []}
    cur = None
    for line in out.splitlines():
        if line.startswith("@@"):
            cur = line[2:]
            continue
        if cur in sections and line.strip():
            sections[cur].append(line)

    now = _num(sections["NOW"][0]) if sections["NOW"] else 0
    boot = _num(sections["BOOT"][0]) if sections["BOOT"] else 0
    hz = _num(sections["HZ"][0]) if sections["HZ"] else 0

    # `date` and /proc/stat run before the pane loop and cannot fail on a working
    # box, so their absence means the script broke partway. That has to be a
    # failed snapshot, not an empty one: an empty snapshot is how this backend
    # says every job ended, and a missing boot time would also collapse every
    # job id to `pid_0` and merge two reruns' history under one key.
    if not now or not boot or not hz:
        first = next((l for l in err.splitlines() if l.strip()), "")
        return Snapshot(cl.name, False, error=(
            "remote script gave no clock or boot time — it died partway"
            + (f": {first[:120]}" if first else " (is /proc readable?)")))

    gpus, apps = parse_gpu_inventory(sections["GPU"], sections["APPS"])

    jobs = []
    for line in sections["JOBS"]:
        f = line.split(FS)
        if len(f) != JOB_FIELDS:
            continue
        (sess, wi, pi, wname, _pane, _ppid, jp, ticks, et, comm, cmd,
         pidstr, log, mtime, errs, series, lastline) = f
        # pid alone would be reused inside the 90-day retention and silently
        # concatenate two runs' loss curves. Start time makes the pair unique.
        start = boot + _num(ticks) // hz
        jid = f"{jp.strip()}_{start}"
        pids = pidstr.split()
        gpu, cards = gpu_for(pids, gpus, apps) if want_gpu else (None, "")
        j = Job(
            cluster=cl.name, jid=jid, partition=sess.strip(),
            name=job_name(wname, comm, cmd), state="R",
            elapsed=hms(_num(et)), timeleft="UNLIMITED", real=jp.strip(), reason="",
            node=" ".join(x for x in
                          (f"{sess.strip()}:{wi.strip()}.{pi.strip()}", cards) if x),
        )
        j.gpu = gpu
        j.gpu_err = "" if gpu else "cpu only"
        j.log = os.path.basename(log) or None
        j.mtime = _num(mtime)
        j.errors = _num(errs)
        j.series = [float(v) for v in series.split(";") if v.strip()][-st.metric_points:]
        j.lastline = lastline
        jobs.append(j)

    # Connected fine but the remote script complained — same as the Slurm path,
    # do not render that as "nothing running".
    warn = "" if jobs or not err else err.splitlines()[0][:120]
    return Snapshot(cl.name, True, jobs, [], now, warn=warn,
                    cpu_watts=parse_cpu_watts(sections["CPU"]))


# --------------------------------------------------------------- log tail --

JID_RE = re.compile(r"^([0-9]+)_[0-9]+$")


def log_tail(cl: Cluster, jid: str, lines: int = 80) -> tuple[str, str]:
    """Last N lines of a running job, from its log file or its pane buffer.

    Only the pid half of the job id reaches the remote shell, and only after it
    has been matched as digits.
    """
    m = JID_RE.match(jid)
    if not m:
        return "", "(refusing a job id that is not <pid>_<start>)"
    pid = m.group(1)
    n = max(1, min(500, int(lines)))
    script = (
        'jp=' + pid + '\n'
        '[ -d /proc/$jp ] || { echo "@@FILE (gone)"; exit 0; }\n'
        'fd1=$(readlink /proc/$jp/fd/1 2>/dev/null)\n'
        'case "$fd1" in\n'
        '  ""|/dev/*|socket:*|pipe:*|anon_inode:*) fd1="" ;;\n'
        '  *) [ -f "$fd1" ] || fd1="" ;;\n'
        'esac\n'
        'if [ -n "$fd1" ]; then\n'
        '  echo "@@FILE $fd1"\n'
        '  tail -n ' + str(n) + ' "$fd1" 2>/dev/null | tr -d "\\r"\n'
        '  exit 0\n'
        'fi\n'
        # No log file, so walk up to the pane that owns this pid and read its
        # scrollback instead.
        'cur=$jp\n'
        'pane=""\n'
        'for _ in 1 2 3 4 5 6 7 8; do\n'
        '  pane=$(tmux list-panes -a -F "#{pane_pid} #{pane_id}" 2>/dev/null '
        '| awk -v p="$cur" "\\$1==p{print \\$2}")\n'
        '  [ -n "$pane" ] && break\n'
        '  cur=$(sed "s/.*) //" /proc/$cur/stat 2>/dev/null | awk "{print \\$2}")\n'
        '  case "$cur" in ""|0|1) break ;; esac\n'
        'done\n'
        '[ -z "$pane" ] && { echo "@@FILE (no log file, and no pane found)"; exit 0; }\n'
        'echo "@@FILE (pane $pane scrollback)"\n'
        'tmux capture-pane -p -J -S -' + str(n) + ' -t "$pane" 2>/dev/null | tr -d "\\r"\n'
    )
    res = ssh(cl.host, script, timeout=cl.command_timeout, opts=ssh_opts_for(cl))
    if not res.ok:
        return "", f"({res.reason})"
    out = res.out.splitlines()
    path = ""
    for i, line in enumerate(out):
        if line.startswith("@@FILE"):
            path = line[7:].strip()
            out = out[i + 1:]
            break
    return path, "\n".join(out)
