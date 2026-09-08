"""Alert rules, returned structured so the store can key them per job.

Same rules the terminal dashboard applies in collector.alerts_for, but each
alert carries its cluster and job id instead of baking them into a string.
"""
from __future__ import annotations

from statistics import median

from collector import minutes_left


def alerts_for(snap, st) -> list:
    """-> [(kind, msg, cluster, jid)]"""
    out = []
    for j in snap.jobs:
        if j.state != "R":
            continue
        tag = f"{snap.cluster}/{j.jid} {j.name.strip()[:28]}"
        add = lambda kind, msg: out.append((kind, msg, snap.cluster, j.jid))

        if j.log and j.mtime:
            idle_min = (snap.remote_now - j.mtime) / 60
            if idle_min > st.stall_minutes:
                add("STALL", f"{tag}: log untouched for {idle_min:.0f} min")

        if j.series:
            last = j.series[-1]
            if last != last or last in (float("inf"), float("-inf")):
                add("NAN", f"{tag}: loss is NaN/inf")
            elif len(j.series) > 8:
                # Median of the early window, not its minimum. A single small
                # outlier near the top of a log — a per-component loss, a warmup
                # line — used to drag the baseline down far enough that every
                # healthy step afterwards read as a 10x divergence.
                base = median(j.series[: max(3, len(j.series) // 4)])
                if base > 0 and last > 10 * base:
                    add("DIVERGE", f"{tag}: loss {last:.4g} is 10x its early median {base:.4g}")

        if j.errors:
            add("ERR", f"{tag}: {j.errors} OOM/CUDA/traceback hits in the log tail")

        ml = minutes_left(j.timeleft)
        if ml is not None and ml < st.timeleft_warn_min:
            add("WALL", f"{tag}: {ml:.0f} min of walltime left")

        if j.gpu and j.gpu[2] < st.low_util_pct:
            add("IDLE", f"{tag}: GPU at {j.gpu[2]}% util")
    return out
