"""Read-side aggregations for the web UI.

Everything here reads the SQLite tables the poller fills. GPU-hours are
integrated from the gap between consecutive samples of the same job, capped at
CAP_S so a service outage does not get billed as continuous GPU time.
"""
from __future__ import annotations

import json
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


# ------------------------------------------------------------- energy ----

def energy(conn, days: float = 7, price_per_kwh: float = 0.12) -> dict:
    """Energy drawn by GPUs, per cluster and per job.

    Same capped-gap integration as gpu_hours: a sample's wattage is held over
    the interval since the previous sample, and a gap longer than CAP_S is
    treated as downtime rather than billed.

    Measured and estimated are returned apart. TDP x util is a rough proxy that
    reads low at low utilization -- an idle A100 still pulls 40-60 W -- so
    adding it into a measured total would quietly corrupt the number.
    """
    since = _since(days)
    acc: dict = {}

    def add(cluster, jid, kwh, est_kwh):
        d = acc.setdefault((cluster, jid), {
            "cluster": cluster, "jid": jid, "name": "", "kwh": 0.0,
            "est_kwh": 0.0, "watts": None, "est_watts": None,
            "gpu_name": "", "scope": "job"})
        d["kwh"] += kwh
        d["est_kwh"] += est_kwh

    for r in conn.execute(
        """
        WITH s AS (
          SELECT cluster, jid, ts, power_w, power_est_w,
                 ts - LAG(ts) OVER (PARTITION BY cluster, jid ORDER BY ts) AS dt
          FROM samples
          WHERE ts>=? AND (power_w IS NOT NULL OR power_est_w IS NOT NULL)
        )
        SELECT cluster, jid,
               SUM(MIN(dt, ?) * COALESCE(power_w, 0)) / 3600.0 / 1000.0 AS kwh,
               SUM(MIN(dt, ?) * COALESCE(power_est_w, 0)) / 3600.0 / 1000.0 AS est_kwh
        FROM s WHERE dt IS NOT NULL AND dt <= ?
        GROUP BY cluster, jid
        """, (since, CAP_S, CAP_S, CAP_S)):
        add(r["cluster"], r["jid"], r["kwh"] or 0, r["est_kwh"] or 0)

    for r in conn.execute(
        "SELECT cluster, jid, "
        "       SUM(dt_s * COALESCE(power_w, 0))/3600.0/1000.0 AS kwh, "
        "       SUM(dt_s * COALESCE(power_est_w, 0))/3600.0/1000.0 AS est_kwh "
        "FROM samples_hourly WHERE hour>=? "
        "  AND (power_w IS NOT NULL OR power_est_w IS NOT NULL) "
        "GROUP BY cluster, jid", (since,)):
        add(r["cluster"], r["jid"], r["kwh"] or 0, r["est_kwh"] or 0)

    # Live draw comes off the newest sample rather than the integration, so the
    # panel shows what the cards are pulling now, not an average. The row is
    # named by an explicit correlated MAX rather than by bare columns beside an
    # aggregate, which only happens to work in SQLite.
    for r in conn.execute(
        "SELECT cluster, jid, ts, power_w, power_est_w, gpu_name, gpu_scope "
        "FROM samples s WHERE ts>=? AND state='R' AND ts = ("
        "  SELECT MAX(ts) FROM samples i "
        "   WHERE i.cluster=s.cluster AND i.jid=s.jid AND i.state='R')",
        (_since(0.02),)):
        d = acc.get((r["cluster"], r["jid"]))
        if d is not None:
            d["watts"] = r["power_w"]
            d["est_watts"] = r["power_est_w"]
            d["gpu_name"] = r["gpu_name"] or ""
            d["scope"] = r["gpu_scope"] or "job"

    names = {(r["cluster"], r["jid"]): r["name"] or ""
             for r in conn.execute("SELECT cluster, jid, name FROM jobs")}

    jobs, clusters = [], {}
    for key, d in acc.items():
        if d["kwh"] <= 0 and d["est_kwh"] <= 0:
            continue
        d["name"] = names.get(key, "")
        d["dollars"] = round((d["kwh"] or d["est_kwh"]) * price_per_kwh, 2)
        jobs.append(d)
        c = clusters.setdefault(d["cluster"], {
            "cluster": d["cluster"], "kwh": 0.0, "est_kwh": 0.0,
            "watts": 0.0, "est_watts": 0.0, "jobs": 0, "node_scoped": 0})
        c["kwh"] += d["kwh"]
        c["est_kwh"] += d["est_kwh"]
        c["watts"] += d.get("watts") or 0
        c["est_watts"] += d.get("est_watts") or 0
        c["jobs"] += 1
        c["node_scoped"] += 1 if d.get("scope") == "node" else 0

    for c in clusters.values():
        for k in ("kwh", "est_kwh"):
            c[k] = round(c[k], 3)
        for k in ("watts", "est_watts"):
            c[k] = round(c[k])
        c["dollars"] = round((c["kwh"] or c["est_kwh"]) * price_per_kwh, 2)

    for d in jobs:
        d["kwh"] = round(d["kwh"], 3)
        d["est_kwh"] = round(d["est_kwh"], 3)

    return {"price_per_kwh": price_per_kwh,
            "clusters": sorted(clusters.values(), key=lambda c: -c["kwh"]),
            "jobs": sorted(jobs, key=lambda d: -(d["kwh"] or d["est_kwh"]))}


# ------------------------------------------------------ cluster capacity ----

CLUSTER_COLS = ("running", "pending", "other", "nodes_total", "nodes_alloc",
                "nodes_idle", "nodes_mixed", "nodes_down", "cpus_alloc",
                "cpus_idle", "cpus_total", "gpus_total", "gpus_alloc")


def cluster_stats(conn, days: float = 1, bucket_s: int = 300) -> dict:
    """Cluster-wide queue and node state: where each partition stands now, and
    how the whole cluster's load moved over the window.

    "ALL" is the synthetic whole-cluster partition and is pulled out of the
    per-partition list, since a node in three partitions is counted in each and
    the partition rows therefore do not sum to it.
    """
    since = _since(days)
    latest = {}
    for r in conn.execute(
        "SELECT c.* FROM cluster_samples c JOIN ("
        "  SELECT cluster, partition, MAX(ts) AS ts FROM cluster_samples "
        "  WHERE ts>=? GROUP BY cluster, partition) m "
        "ON m.cluster=c.cluster AND m.partition=c.partition AND m.ts=c.ts",
        (since,)):
        row = {k: r[k] for k in ("ts", "partition", "n", *CLUSTER_COLS)}
        row["gpu_models"] = json.loads(r["gpu_models"] or "{}")
        d = latest.setdefault(r["cluster"], {"cluster": r["cluster"],
                                             "total": None, "partitions": []})
        if r["partition"] == "ALL":
            d["total"] = row
        else:
            d["partitions"].append(row)

    for d in latest.values():
        d["partitions"].sort(key=lambda p: (-p["running"], p["partition"]))

    series: dict = {}
    for r in conn.execute(
        # *1.0 forces real division. Both operands are INTEGER, and SQLite would
        # otherwise truncate a mean queue depth of 22.7 to 22.
        "SELECT cluster, ts/? AS b, "
        "       SUM(running*n)*1.0/SUM(n) AS running, "
        "       SUM(pending*n)*1.0/SUM(n) AS pending, "
        "       SUM(gpus_alloc*n)*1.0/SUM(n) AS gpus_alloc, "
        "       SUM(gpus_total*n)*1.0/SUM(n) AS gpus_total "
        "FROM cluster_samples WHERE ts>=? AND partition='ALL' "
        "GROUP BY cluster, b ORDER BY b", (bucket_s, since)):
        d = series.setdefault(r["cluster"], {"ts": [], "running": [], "pending": [],
                                             "gpus_alloc": [], "gpus_total": []})
        d["ts"].append(r["b"] * bucket_s)
        for k in ("running", "pending", "gpus_alloc", "gpus_total"):
            d[k].append(round(r[k] or 0, 1))

    for name, d in latest.items():
        d["series"] = series.get(name, {"ts": [], "running": [], "pending": [],
                                        "gpus_alloc": [], "gpus_total": []})
    return latest


# ------------------------------------------------- memory efficiency ----

MEM_UNITS = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}


def parse_mem_mib(raw: str) -> float | None:
    """sacct memory ("56G", "39452548K", "0") -> MiB.

    ReqMem on Slurm 21.08+ is a plain size; older versions append `n` or `c`
    for per-node / per-cpu, which is stripped and ignored -- reporting the base
    figure beats reporting a number scaled by a cpu count we do not have.
    """
    t = (raw or "").strip().rstrip("nc")
    if not t:
        return None
    unit = t[-1].upper()
    if unit in MEM_UNITS:
        t = t[:-1]
    else:
        unit = "M"
    try:
        v = float(t)
    except ValueError:
        return None
    return v * MEM_UNITS[unit] if v else None


def mem_efficiency(conn, days: float = 30) -> list:
    """Per finished job: memory asked for against the peak actually used.

    MaxRSS lives on the .batch step and ReqMem on the allocation line, so both
    arrive from the collector's separate memory query. Jobs that finished before
    those columns existed have neither and are left out rather than shown at 0%.
    """
    out = []
    for r in conn.execute(
        "SELECT cluster, jid, name, state, req_mem, max_rss, end_ts "
        "FROM finished WHERE COALESCE(first_written, end_ts) >= ? "
        "  AND req_mem IS NOT NULL AND max_rss IS NOT NULL",
        (_since(days),)):
        req, used = parse_mem_mib(r["req_mem"]), parse_mem_mib(r["max_rss"])
        if not req or used is None:
            continue
        out.append({"cluster": r["cluster"], "jid": r["jid"],
                    "name": r["name"] or "", "state": r["state"],
                    "end_ts": r["end_ts"],
                    "req_mib": round(req), "used_mib": round(used),
                    "pct": round(used / req * 100, 1)})
    return sorted(out, key=lambda d: d["pct"])
