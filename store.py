"""SQLite persistence for jobdash.

The terminal dashboard threw everything away between refreshes. This keeps a
real time series so the web UI can chart GPU utilization, loss, queue waits,
and finished-job outcomes long after sacct's window has rolled past them.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from collector import minutes_left

DB_PATH = Path(__file__).with_name("jobdash.db")
GAP_CAP_S = 300   # a gap longer than this is downtime, not GPU time

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  ts INTEGER NOT NULL, cluster TEXT NOT NULL, jid TEXT NOT NULL,
  state TEXT, gpu_used INTEGER, gpu_total INTEGER, util INTEGER, gpu_n INTEGER,
  loss REAL, log_mtime INTEGER, errors INTEGER
);
CREATE INDEX IF NOT EXISTS ix_samples_job ON samples(cluster, jid, ts);
CREATE INDEX IF NOT EXISTS ix_samples_ts  ON samples(ts);

CREATE TABLE IF NOT EXISTS jobs (
  cluster TEXT NOT NULL, jid TEXT NOT NULL,
  name TEXT, partition TEXT, node TEXT,
  first_seen INTEGER, last_seen INTEGER,
  pend_start INTEGER, run_start INTEGER,
  last_state TEXT, log TEXT,
  PRIMARY KEY (cluster, jid)
);

CREATE TABLE IF NOT EXISTS finished (
  cluster TEXT NOT NULL, jid TEXT NOT NULL,
  name TEXT, state TEXT, elapsed TEXT, exitcode TEXT,
  end_ts INTEGER, end_raw TEXT, maxrss TEXT,
  submit_ts INTEGER, start_ts INTEGER, first_written INTEGER,
  PRIMARY KEY (cluster, jid)
);
CREATE INDEX IF NOT EXISTS ix_finished_end ON finished(end_ts);

CREATE TABLE IF NOT EXISTS samples_hourly (
  hour INTEGER NOT NULL, cluster TEXT NOT NULL, jid TEXT NOT NULL,
  util REAL, gpu_used INTEGER, gpu_n INTEGER, loss REAL,
  n INTEGER, dt_s INTEGER,
  PRIMARY KEY (hour, cluster, jid)
);
CREATE INDEX IF NOT EXISTS ix_hourly_hour ON samples_hourly(hour);

CREATE TABLE IF NOT EXISTS alerts (
  key TEXT PRIMARY KEY, kind TEXT, cluster TEXT, jid TEXT, msg TEXT,
  first_seen INTEGER, last_seen INTEGER, acked INTEGER DEFAULT 0,
  cleared_at INTEGER
);
CREATE INDEX IF NOT EXISTS ix_alerts_last ON alerts(last_seen);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""

_lock = threading.Lock()


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    c = sqlite3.connect(path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


ADDED_COLUMNS = {
    "finished": {"submit_ts": "INTEGER", "start_ts": "INTEGER",
                 "first_written": "INTEGER"},
}

# Indexes over added columns run after the ALTERs, never inside SCHEMA.
POST_MIGRATE = """
CREATE INDEX IF NOT EXISTS ix_finished_written ON finished(first_written);
"""

# Rows written before first_written existed would fail every windowed query,
# so date them by their end time and fall back to the migration moment.
BACKFILL_WRITTEN = (
    "UPDATE finished SET first_written = COALESCE(end_ts, ?) "
    "WHERE first_written IS NULL")


def init(conn: sqlite3.Connection) -> None:
    with _lock:
        conn.executescript(SCHEMA)
        for table, cols in ADDED_COLUMNS.items():
            have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols.items():
                if name not in have:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        conn.executescript(POST_MIGRATE)
        conn.execute(BACKFILL_WRITTEN, (int(time.time()),))
        conn.commit()


# ------------------------------------------------------------------ writes --

def parse_end(raw: str) -> int | None:
    """sacct End/Submit/Start are ISO-ish local time, or 'Unknown'/'None'."""
    raw = (raw or "").strip()
    if not raw or raw in ("Unknown", "None"):
        return None
    try:
        return int(time.mktime(time.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")))
    except ValueError:
        return None


def run_start(ts: int, j) -> int | None:
    """Real start time, not first-sighting. Slurm reports elapsed walltime, so a
    job already running when the service came up lands on the timeline at the
    hour it actually started.

    Unparsable elapsed stays NULL rather than defaulting to now — the Gantt
    already skips a missing run_start, and a wrong bar is worse than no bar.
    `minutes_left` parses the same D-HH:MM:SS family, sentinels aside.
    """
    if j.state != "R":
        return None
    mins = minutes_left(j.elapsed)
    return ts - int(mins * 60) if mins is not None else None


def write_snapshot(conn: sqlite3.Connection, snap) -> None:
    """One poller tick for one cluster. Skipped entirely if the ssh failed, so
    an unreachable cluster leaves a gap rather than a row of zeros.
    """
    if not snap.ok:
        return
    ts = snap.remote_now or int(time.time())
    with _lock:
        for j in snap.jobs:
            g = j.gpu or (None, None, None, None)
            conn.execute(
                "INSERT INTO samples (ts,cluster,jid,state,gpu_used,gpu_total,util,gpu_n,"
                "loss,log_mtime,errors) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (ts, snap.cluster, j.jid, j.state, g[0], g[1], g[2], g[3],
                 j.series[-1] if j.series else None, j.mtime or None, j.errors))
            conn.execute(
                "INSERT INTO jobs (cluster,jid,name,partition,node,first_seen,last_seen,"
                "pend_start,run_start,last_state,log) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(cluster,jid) DO UPDATE SET "
                "  last_seen=excluded.last_seen, last_state=excluded.last_state,"
                "  name=excluded.name, node=COALESCE(NULLIF(excluded.node,''), jobs.node),"
                "  log=COALESCE(excluded.log, jobs.log),"
                "  pend_start=COALESCE(jobs.pend_start, excluded.pend_start),"
                "  run_start=COALESCE(jobs.run_start, excluded.run_start)",
                (snap.cluster, j.jid, j.name.strip(), j.partition.strip(), j.node,
                 ts, ts,
                 ts if j.state == "PD" else None,
                 run_start(ts, j),
                 j.state, j.log))

        for f in snap.finished:
            row = (list(f) + [""] * 9)[:9]
            jid, name, state, elapsed, exitcode, end, maxrss, submit, start = row
            jid = jid.split(".")[0].strip()
            if not jid:
                continue

            # first_written stamps when we learned of the job. `end_ts` is NULL
            # for a still-running job and for any End sacct would not parse, so
            # windowed queries need a timestamp that is always present.
            conn.execute(
                "INSERT INTO finished (cluster,jid,name,state,elapsed,exitcode,end_ts,"
                "end_raw,maxrss,submit_ts,start_ts,first_written) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(cluster,jid) DO UPDATE SET state=excluded.state,"
                "  elapsed=excluded.elapsed, exitcode=excluded.exitcode,"
                "  end_ts=COALESCE(excluded.end_ts, finished.end_ts),"
                "  end_raw=excluded.end_raw, maxrss=excluded.maxrss,"
                "  submit_ts=COALESCE(excluded.submit_ts, finished.submit_ts),"
                "  start_ts=COALESCE(excluded.start_ts, finished.start_ts),"
                "  first_written=COALESCE(finished.first_written, excluded.first_written)",
                (snap.cluster, jid, name.strip(), state.strip(), elapsed.strip(),
                 exitcode.strip(), parse_end(end), end.strip(), maxrss.strip(),
                 parse_end(submit), parse_end(start), ts))
        conn.commit()


def sync_alerts(conn: sqlite3.Connection, active: list) -> list:
    """Reconcile the live alert list against the table. Returns the alerts that
    are newly firing this tick, so notification is once-per-onset rather than
    once-per-refresh. An alert that clears and later recurs re-arms.
    """
    ts = int(time.time())
    fresh = []
    with _lock:
        live_keys = set()
        for kind, msg, cluster, jid in active:
            key = f"{cluster}/{jid}/{kind}"
            live_keys.add(key)
            row = conn.execute(
                "SELECT last_seen, cleared_at FROM alerts WHERE key=?", (key,)).fetchone()
            if row is None or row["cleared_at"] is not None:
                fresh.append((kind, msg))
                conn.execute(
                    "INSERT INTO alerts (key,kind,cluster,jid,msg,first_seen,last_seen,"
                    "acked,cleared_at) VALUES (?,?,?,?,?,?,?,0,NULL) "
                    "ON CONFLICT(key) DO UPDATE SET msg=excluded.msg,"
                    "  first_seen=excluded.first_seen, last_seen=excluded.last_seen,"
                    "  acked=0, cleared_at=NULL",
                    (key, kind, cluster, jid, msg, ts, ts))
            else:
                conn.execute("UPDATE alerts SET last_seen=?, msg=? WHERE key=?",
                             (ts, msg, key))

        stale = [r["key"] for r in conn.execute(
            "SELECT key FROM alerts WHERE cleared_at IS NULL")]
        for key in stale:
            if key not in live_keys:
                conn.execute("UPDATE alerts SET cleared_at=? WHERE key=?", (ts, key))
        conn.commit()
    return fresh


def ack_alert(conn: sqlite3.Connection, key: str) -> None:
    with _lock:
        conn.execute("UPDATE alerts SET acked=1 WHERE key=?", (key,))
        conn.commit()


def backfill_jsonl(conn: sqlite3.Connection, path: Path) -> int:
    """One-shot import of the old append-only history file. It only ever held a
    last-loss value, so it fills the loss column and nothing else.
    """
    if not path.exists():
        return 0
    with _lock:
        done = conn.execute("SELECT v FROM meta WHERE k='backfill'").fetchone()
        if done:
            return 0
        n = 0
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            conn.execute(
                "INSERT INTO samples (ts,cluster,jid,state,loss) VALUES (?,?,?,'R',?)",
                (int(d["t"]), d["cluster"], str(d["jid"]), float(d["last"])))
            n += 1
        conn.execute("INSERT OR REPLACE INTO meta (k,v) VALUES ('backfill',?)",
                     (str(int(time.time())),))
        conn.commit()
    return n


RAW_KEEP_DAYS = 3     # raw samples past this age live on only in samples_hourly


def rollup(conn: sqlite3.Connection, keep_days: int = RAW_KEEP_DAYS) -> int:
    """Fold raw samples older than keep_days into one row per job per hour.

    `dt_s` carries the same capped-gap integration gpu_hours does on raw rows,
    so GPU-hours stay comparable across the boundary. Only whole hours that are
    already closed get folded, so a partially-written hour is never halved.
    """
    cutoff = int(time.time()) - keep_days * 86400
    cutoff -= cutoff % 3600
    with _lock:
        conn.execute(
            """
            INSERT INTO samples_hourly (hour,cluster,jid,util,gpu_used,gpu_n,loss,n,dt_s)
            SELECT ts/3600*3600 AS hour, cluster, jid,
                   AVG(util), MAX(gpu_used), MAX(gpu_n), AVG(loss), COUNT(*),
                   COALESCE(SUM(MIN(dt, ?)), 0)
            FROM (
              SELECT ts, cluster, jid, util, gpu_used, gpu_n, loss,
                     ts - LAG(ts) OVER (PARTITION BY cluster, jid ORDER BY ts) AS dt
              FROM samples WHERE ts < ?
            )
            GROUP BY hour, cluster, jid
            ON CONFLICT(hour,cluster,jid) DO UPDATE SET
              util = CASE WHEN samples_hourly.util IS NULL THEN excluded.util
                          WHEN excluded.util IS NULL THEN samples_hourly.util
                          ELSE (samples_hourly.util * samples_hourly.n
                                + excluded.util * excluded.n)
                               / (samples_hourly.n + excluded.n) END,
              loss = CASE WHEN samples_hourly.loss IS NULL THEN excluded.loss
                          WHEN excluded.loss IS NULL THEN samples_hourly.loss
                          ELSE (samples_hourly.loss * samples_hourly.n
                                + excluded.loss * excluded.n)
                               / (samples_hourly.n + excluded.n) END,
              gpu_used = MAX(COALESCE(samples_hourly.gpu_used, 0),
                             COALESCE(excluded.gpu_used, 0)),
              gpu_n    = MAX(COALESCE(samples_hourly.gpu_n, 0),
                             COALESCE(excluded.gpu_n, 0)),
              n        = samples_hourly.n + excluded.n,
              dt_s     = samples_hourly.dt_s + excluded.dt_s
            """, (GAP_CAP_S, cutoff))
        n = conn.execute("SELECT changes() c").fetchone()["c"]
        conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
        conn.commit()
    return n


def prune(conn: sqlite3.Connection, keep_days: int = 90) -> None:
    cutoff = int(time.time()) - keep_days * 86400
    with _lock:
        conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM samples_hourly WHERE hour < ?", (cutoff,))
        conn.execute("DELETE FROM alerts WHERE cleared_at IS NOT NULL AND cleared_at < ?",
                     (cutoff,))
        conn.commit()
