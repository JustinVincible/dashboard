"""Read-side aggregations for the web UI.

Everything here reads the SQLite tables the poller fills. GPU-hours are
integrated from the gap between consecutive samples of the same job, capped at
CAP_S so a service outage does not get billed as continuous GPU time.
"""
from __future__ import annotations

import time

from store import GAP_CAP_S as CAP_S


def _since(days: float) -> int:
    return int(time.time() - days * 86400)


def job_series(conn, cluster: str, jid: str, days: float = 7) -> dict:
    rows = conn.execute(
        "SELECT ts, util, gpu_used, gpu_total, loss FROM samples "
        "WHERE cluster=? AND jid=? AND ts>=? ORDER BY ts", (cluster, jid, _since(days))
    ).fetchall()
    return {
        "ts":   [r["ts"] for r in rows],
        "util": [r["util"] for r in rows],
        "mem":  [r["gpu_used"] for r in rows],
        "memtotal": [r["gpu_total"] for r in rows],
        "loss": [r["loss"] for r in rows],
    }


def cluster_util(conn, days: float = 1, bucket_s: int = 300) -> dict:
    """Mean GPU utilization and mean concurrent GPU count per cluster, bucketed.

    `gpus` and `mem` divide by the distinct sample timestamps in the bucket, so
    they read as "GPUs held at once" rather than "GPUs summed over every poll" —
    the latter scales with the poll rate and means nothing.

    Reads raw samples for the recent window and the hourly rollup behind it, so
    a 30-day range still charts after raw rows have been folded away.
    """
    since = _since(days)
    out: dict = {}

    def put(cluster, ts, util, gpus, mem, jobs):
        d = out.setdefault(cluster, {"ts": [], "util": [], "gpus": [], "mem": [], "jobs": []})
        d["ts"].append(ts)
        d["util"].append(round(util, 1))
        d["gpus"].append(round(gpus, 1))
        d["mem"].append(round(mem))
        d["jobs"].append(jobs)

    for r in conn.execute(
        "SELECT cluster, ts/? AS b, AVG(util) AS util, "
        "       SUM(gpu_n)*1.0/COUNT(DISTINCT ts) AS gpus, "
        "       SUM(gpu_used)*1.0/COUNT(DISTINCT ts) AS mem, "
        "       COUNT(DISTINCT jid) AS jobs "
        "FROM samples WHERE ts>=? AND util IS NOT NULL "
        "GROUP BY cluster, b ORDER BY b", (bucket_s, since)):
        put(r["cluster"], r["b"] * bucket_s, r["util"], r["gpus"] or 0, r["mem"] or 0, r["jobs"])

    # Hourly rows are already one per job per hour, so the divisor is the number
    # of hours the bucket spans rather than a sample count.
    hours = max(1, bucket_s // 3600)
    covered = {(c, t) for c, d in out.items() for t in d["ts"]}
    for r in conn.execute(
        "SELECT cluster, hour/? AS b, AVG(util) AS util, "
        "       SUM(gpu_n)*1.0 AS gpus, SUM(gpu_used)*1.0 AS mem, "
        "       COUNT(DISTINCT jid) AS jobs "
        "FROM samples_hourly WHERE hour>=? AND util IS NOT NULL "
        "GROUP BY cluster, b ORDER BY b", (bucket_s, since)):
        ts = r["b"] * bucket_s
        if (r["cluster"], ts) in covered:
            continue
        put(r["cluster"], ts, r["util"], (r["gpus"] or 0) / hours,
            (r["mem"] or 0) / hours, r["jobs"])

    for d in out.values():
        order = sorted(range(len(d["ts"])), key=lambda i: d["ts"][i])
        for k in d:
            d[k] = [d[k][i] for i in order]
    return out


def gpu_hours(conn, days: float = 30) -> list:
    """Per job: GPU-hours held, and how many of them ran below useful load.

    The "wasted" figure is GPU-hours x (1 - mean util). It is the number worth
    staring at — it converts a vague "that run looked idle" into hours.

    Raw samples cover the recent window; samples_hourly covers everything older,
    carrying the same capped-gap `dt_s` so the two add up.
    """
    since = _since(days)
    acc: dict = {}

    def add(cluster, jid, gpu_h, wasted_h, util_w, w, last_ts):
        d = acc.setdefault((cluster, jid), {"cluster": cluster, "jid": jid, "name": "",
                                            "gpu_h": 0.0, "wasted_h": 0.0,
                                            "_uw": 0.0, "_w": 0.0, "last_ts": 0})
        d["gpu_h"] += gpu_h
        d["wasted_h"] += wasted_h
        d["_uw"] += util_w
        d["_w"] += w
        d["last_ts"] = max(d["last_ts"], last_ts)

    for r in conn.execute(
        """
        WITH s AS (
          SELECT cluster, jid, ts, util, gpu_n,
                 ts - LAG(ts) OVER (PARTITION BY cluster, jid ORDER BY ts) AS dt
          FROM samples WHERE ts>=? AND gpu_n IS NOT NULL
        )
        SELECT cluster, jid,
               SUM(MIN(dt, ?) * gpu_n) / 3600.0 AS gpu_h,
               SUM(MIN(dt, ?) * gpu_n * (100 - util) / 100.0) / 3600.0 AS wasted_h,
               SUM(util) AS util_sum, COUNT(*) AS n, MAX(ts) AS last_ts
        FROM s WHERE dt IS NOT NULL AND dt <= ?
        GROUP BY cluster, jid
        """, (since, CAP_S, CAP_S, CAP_S)):
        add(r["cluster"], r["jid"], r["gpu_h"] or 0, r["wasted_h"] or 0,
            r["util_sum"] or 0, r["n"], r["last_ts"])

    for r in conn.execute(
        "SELECT cluster, jid, SUM(dt_s * gpu_n)/3600.0 AS gpu_h, "
        "       SUM(dt_s * gpu_n * (100 - util) / 100.0)/3600.0 AS wasted_h, "
        "       SUM(util * n) AS util_sum, SUM(n) AS n, MAX(hour) AS last_ts "
        "FROM samples_hourly WHERE hour>=? AND gpu_n IS NOT NULL AND util IS NOT NULL "
        "GROUP BY cluster, jid", (since,)):
        add(r["cluster"], r["jid"], r["gpu_h"] or 0, r["wasted_h"] or 0,
            r["util_sum"] or 0, r["n"], r["last_ts"])

    names = {(r["cluster"], r["jid"]): r["name"] or ""
             for r in conn.execute("SELECT cluster, jid, name FROM jobs")}
    out = []
    for key, d in acc.items():
        if d["gpu_h"] <= 0:
            continue
        d["name"] = names.get(key, "")
        d["mean_util"] = d["_uw"] / d["_w"] if d["_w"] else 0.0
        del d["_uw"], d["_w"]
        out.append(d)
    return sorted(out, key=lambda d: -d["wasted_h"])


def queue_waits(conn, days: float = 30) -> list:
    """Median and p90 pend time by cluster+partition.

    sacct's Submit and Start are authoritative and cover jobs that flipped from
    pending to running between two polls — array jobs especially, which is why
    this panel used to stay sparse. The observed pend_start/run_start pair is
    the fallback for jobs sacct has not reported yet.
    """
    rows = conn.execute(
        """
        SELECT j.cluster, j.partition,
               (COALESCE(f.start_ts, j.run_start) -
                COALESCE(f.submit_ts, j.pend_start)) / 60.0 AS wait_min
        FROM jobs j LEFT JOIN finished f ON f.cluster=j.cluster AND f.jid=j.jid
        WHERE j.last_seen >= ?
          AND COALESCE(f.submit_ts, j.pend_start) IS NOT NULL
          AND COALESCE(f.start_ts, j.run_start) IS NOT NULL
          AND COALESCE(f.start_ts, j.run_start) > COALESCE(f.submit_ts, j.pend_start)
        """, (_since(days),)).fetchall()
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["cluster"], r["partition"] or "?"), []).append(r["wait_min"])
    out = []
    for (cluster, part), vals in groups.items():
        vals.sort()
        n = len(vals)
        out.append({"cluster": cluster, "partition": part, "n": n,
                    "median_min": round(vals[n // 2], 1),
                    "p90_min": round(vals[min(n - 1, int(n * 0.9))], 1),
                    "max_min": round(vals[-1], 1)})
    return sorted(out, key=lambda d: -d["median_min"])


def failures(conn, days: float = 30) -> dict:
    """sacct outcomes, plus how many of those jobs had log errors while running."""
    # A NULL end_ts means sacct said Unknown or the End would not parse, which
    # is not a reason to escape the window. first_written bounds those rows.
    rows = conn.execute(
        "SELECT state, COUNT(*) n FROM finished "
        "WHERE COALESCE(end_ts, first_written, 0) >= ? "
        "GROUP BY state ORDER BY n DESC", (_since(days),)).fetchall()
    err = conn.execute(
        "SELECT COUNT(DISTINCT cluster||'/'||jid) n FROM samples "
        "WHERE ts>=? AND errors > 0", (_since(days),)).fetchone()["n"]
    return {"states": [dict(r) for r in rows], "jobs_with_log_errors": err}


def recent_finished(conn, days: float = 7, limit: int = 60) -> list:
    rows = conn.execute(
        "SELECT * FROM finished WHERE COALESCE(end_ts, first_written, 0) >= ? "
        "ORDER BY COALESCE(end_ts, first_written, 0) DESC LIMIT ?",
        (_since(days), limit)).fetchall()
    return [dict(r) for r in rows]


def timeline(conn, days: float = 7) -> list:
    """Per job: when it was first seen queued, when it started, when last seen.

    Feeds the Gantt strip. sacct's Submit/Start win over the observed
    transition, and `end_ts` prefers sacct's real end over last_seen.
    """
    rows = conn.execute(
        "SELECT j.cluster, j.jid, j.name, j.partition, j.first_seen, j.last_seen, "
        "       COALESCE(f.submit_ts, j.pend_start) AS pend_start, "
        "       COALESCE(f.start_ts, j.run_start) AS run_start, "
        "       j.last_state, f.state AS final_state, f.end_ts "
        "FROM jobs j LEFT JOIN finished f ON f.cluster=j.cluster AND f.jid=j.jid "
        "WHERE j.last_seen >= ? ORDER BY j.first_seen", (_since(days),)).fetchall()
    return [dict(r) for r in rows]


def compare_loss(conn, jobs: list, days: float = 30) -> list:
    """Loss curves for several jobs at once, indexed by step rather than clock.

    Runs of different lengths and start times line up when the x axis counts
    samples, which is what comparing a sweep needs. Hourly means fill in behind
    raw retention, so an old run still plots — coarser, but on the same axis.
    """
    since = _since(days)
    out = []
    for cluster, jid in jobs:
        loss = [r["loss"] for r in conn.execute(
            "SELECT loss FROM samples WHERE cluster=? AND jid=? AND ts>=? "
            "AND loss IS NOT NULL ORDER BY ts", (cluster, jid, since))]
        if not loss:
            loss = [r["loss"] for r in conn.execute(
                "SELECT loss FROM samples_hourly WHERE cluster=? AND jid=? AND hour>=? "
                "AND loss IS NOT NULL ORDER BY hour", (cluster, jid, since))]
        name = conn.execute("SELECT name FROM jobs WHERE cluster=? AND jid=?",
                            (cluster, jid)).fetchone()
        out.append({"cluster": cluster, "jid": jid,
                    "name": (name["name"] if name else "") or "",
                    "loss": loss})
    return out


def alert_history(conn, days: float = 14, limit: int = 200) -> list:
    rows = conn.execute(
        "SELECT * FROM alerts WHERE last_seen >= ? ORDER BY last_seen DESC LIMIT ?",
        (_since(days), limit)).fetchall()
    return [dict(r) for r in rows]
