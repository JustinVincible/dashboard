import time

import queries
import store
from conftest import INTERVAL, seed_samples


# ------------------------------------------------------------- cluster_util --

def test_gpu_count_is_concurrency_not_a_sample_sum(conn, now):
    """Two jobs holding 2 GPUs each is 4 GPUs, however often we poll them."""
    bucket = 300
    base = (now - 600) // bucket * bucket
    seed_samples(conn, "gaivi", "1", base, bucket // INTERVAL, util=60, gpu_n=2)
    seed_samples(conn, "gaivi", "2", base, bucket // INTERVAL, util=40, gpu_n=2)

    out = queries.cluster_util(conn, days=1, bucket_s=bucket)["gaivi"]
    assert out["gpus"][0] == 4.0
    assert out["util"][0] == 50.0
    assert out["jobs"][0] == 2


def test_cluster_util_reads_the_rollup_behind_raw_retention(conn, now):
    old_hour = (now - 10 * 86400) // 3600 * 3600
    conn.execute("INSERT INTO samples_hourly (hour,cluster,jid,util,gpu_used,gpu_n,n,dt_s) "
                 "VALUES (?,'gaivi','9',70,8192,4,180,3600)", (old_hour,))
    conn.commit()

    out = queries.cluster_util(conn, days=30, bucket_s=3600)
    assert out["gaivi"]["util"] == [70.0]
    assert out["gaivi"]["gpus"] == [4.0]


# --------------------------------------------------------------- gpu_hours --

def test_gpu_hours_and_waste(conn, now):
    """One hour of wall time on 2 GPUs at 50% util is 2 gpu-h, 1 of them idle."""
    n = 3600 // INTERVAL + 1
    seed_samples(conn, "gaivi", "7", now - 3600, n, util=50, gpu_n=2)

    row = next(r for r in queries.gpu_hours(conn, days=1) if r["jid"] == "7")
    assert abs(row["gpu_h"] - 2.0) < 0.01
    assert abs(row["wasted_h"] - 1.0) < 0.01
    assert abs(row["mean_util"] - 50) < 0.01


def test_gpu_hours_ignores_an_outage_gap(conn, now):
    """A service outage is not billable GPU time — gaps are capped at CAP_S."""
    seed_samples(conn, "gaivi", "8", now - 86400, 2, gpu_n=1)
    seed_samples(conn, "gaivi", "8", now - 60, 2, gpu_n=1)

    row = next(r for r in queries.gpu_hours(conn, days=2) if r["jid"] == "8")
    assert row["gpu_h"] < 2 * queries.CAP_S / 3600.0


def test_gpu_hours_sums_raw_and_rolled_up(conn, now):
    hour = (now - 10 * 86400) // 3600 * 3600
    conn.execute("INSERT INTO samples_hourly (hour,cluster,jid,util,gpu_used,gpu_n,n,dt_s) "
                 "VALUES (?,'gaivi','5',100,8192,1,180,3600)", (hour,))
    conn.commit()
    seed_samples(conn, "gaivi", "5", now - 3600, 3600 // INTERVAL + 1, util=100, gpu_n=1)

    row = next(r for r in queries.gpu_hours(conn, days=30) if r["jid"] == "5")
    assert abs(row["gpu_h"] - 2.0) < 0.02
    assert row["wasted_h"] < 0.01


# ---------------------------------------------------------------- failures --

def test_null_end_ts_does_not_escape_the_window(conn, now):
    old = now - 90 * 86400
    conn.execute("INSERT INTO finished (cluster,jid,state,end_ts,first_written) "
                 "VALUES ('gaivi','old','FAILED',NULL,?)", (old,))
    conn.execute("INSERT INTO finished (cluster,jid,state,end_ts,first_written) "
                 "VALUES ('gaivi','new','COMPLETED',NULL,?)", (now,))
    conn.commit()

    states = {r["state"]: r["n"] for r in queries.failures(conn, days=30)["states"]}
    assert states == {"COMPLETED": 1}


def test_recent_finished_keeps_a_still_running_job(conn, now):
    conn.execute("INSERT INTO finished (cluster,jid,state,end_ts,first_written) "
                 "VALUES ('gaivi','live','RUNNING',NULL,?)", (now,))
    conn.commit()
    assert [r["jid"] for r in queries.recent_finished(conn, days=7)] == ["live"]


# ------------------------------------------------------------- queue_waits --

def test_wait_uses_sacct_when_the_poller_never_saw_the_job_pending(conn, now):
    """The array-job case: submitted and started between two polls."""
    conn.execute("INSERT INTO jobs (cluster,jid,partition,first_seen,last_seen,"
                 "pend_start,run_start) VALUES ('gaivi','a','gpu',?,?,NULL,NULL)",
                 (now - 600, now))
    conn.execute("INSERT INTO finished (cluster,jid,state,submit_ts,start_ts,first_written) "
                 "VALUES ('gaivi','a','COMPLETED',?,?,?)",
                 (now - 1800, now - 1200, now))
    conn.commit()

    row = queries.queue_waits(conn, days=30)[0]
    assert row["partition"] == "gpu"
    assert row["median_min"] == 10.0


def test_wait_falls_back_to_the_observed_transition(conn, now):
    conn.execute("INSERT INTO jobs (cluster,jid,partition,first_seen,last_seen,"
                 "pend_start,run_start) VALUES ('gaivi','b','gpu',?,?,?,?)",
                 (now - 600, now, now - 600, now - 300))
    conn.commit()
    assert queries.queue_waits(conn, days=30)[0]["median_min"] == 5.0


# ---------------------------------------------------------- rollup / prune --

def test_rollup_folds_old_samples_and_preserves_gpu_hours(conn, now):
    old = now - 10 * 86400
    n = 3600 // INTERVAL + 1
    seed_samples(conn, "gaivi", "r", old, n, util=50, gpu_n=2)
    before = next(r for r in queries.gpu_hours(conn, days=30) if r["jid"] == "r")

    store.rollup(conn)

    assert conn.execute("SELECT COUNT(*) c FROM samples").fetchone()["c"] == 0
    after = next(r for r in queries.gpu_hours(conn, days=30) if r["jid"] == "r")
    assert abs(after["gpu_h"] - before["gpu_h"]) < 0.05
    assert abs(after["wasted_h"] - before["wasted_h"]) < 0.05


def test_rollup_leaves_the_recent_window_alone(conn, now):
    seed_samples(conn, "gaivi", "fresh", now - 600, 30)
    store.rollup(conn)
    assert conn.execute("SELECT COUNT(*) c FROM samples").fetchone()["c"] == 30


def test_prune_drops_rolled_up_rows_too(conn, now):
    conn.execute("INSERT INTO samples_hourly (hour,cluster,jid,util,gpu_n,n,dt_s) "
                 "VALUES (?,'gaivi','z',50,1,180,3600)", (now - 200 * 86400,))
    conn.commit()
    store.prune(conn, keep_days=90)
    assert conn.execute("SELECT COUNT(*) c FROM samples_hourly").fetchone()["c"] == 0


# ------------------------------------------------------------------ compare --

def _seed_loss(conn, cluster, jid, values, start_ts, name=""):
    for i, v in enumerate(values):
        conn.execute("INSERT INTO samples (ts,cluster,jid,state,loss) VALUES (?,?,?,'R',?)",
                     (start_ts + i * INTERVAL, cluster, jid, v))
    conn.execute("INSERT OR REPLACE INTO jobs (cluster,jid,name,first_seen,last_seen) "
                 "VALUES (?,?,?,?,?)", (cluster, jid, name, start_ts, start_ts))
    conn.commit()


def test_compare_returns_one_curve_per_job_in_request_order(conn, now):
    _seed_loss(conn, "gaivi", "1", [3.0, 2.0, 1.0], now - 600, "runA")
    _seed_loss(conn, "gaivi", "2", [9.0, 8.0], now - 300, "runB")

    out = queries.compare_loss(conn, [("gaivi", "2"), ("gaivi", "1")], days=1)
    assert [r["jid"] for r in out] == ["2", "1"]
    assert out[0]["loss"] == [9.0, 8.0]
    assert out[1]["name"] == "runA"


def test_compare_falls_back_to_the_rollup_for_an_old_run(conn, now):
    hour = (now - 20 * 86400) // 3600 * 3600
    for i, v in enumerate([5.0, 4.0]):
        conn.execute("INSERT INTO samples_hourly (hour,cluster,jid,loss,n,dt_s) "
                     "VALUES (?,'gaivi','old',?,180,3600)", (hour + i * 3600, v))
    conn.commit()

    out = queries.compare_loss(conn, [("gaivi", "old")], days=30)
    assert out[0]["loss"] == [5.0, 4.0]


def test_compare_of_an_unknown_job_is_empty_not_an_error(conn):
    out = queries.compare_loss(conn, [("gaivi", "nope")], days=30)
    assert out == [{"cluster": "gaivi", "jid": "nope", "name": "", "loss": []}]


def test_rollup_merges_a_second_pass_into_an_existing_hour(conn):
    """A later raw sample landing in an already-folded hour must be merged.
    Skipping the insert would still delete the raw row, losing it from both
    tables."""
    hour = (int(time.time()) - 10 * 86400) // 3600 * 3600
    conn.execute(
        "INSERT INTO samples (ts,cluster,jid,state,util,gpu_used,gpu_n) "
        "VALUES (?,'gaivi','1','R',40,1,1)", (hour + 10,))
    conn.commit()
    store.rollup(conn)

    conn.execute(
        "INSERT INTO samples (ts,cluster,jid,state,util,gpu_used,gpu_n) "
        "VALUES (?,'gaivi','1','R',80,1,1)", (hour + 20,))
    conn.commit()
    store.rollup(conn)

    r = conn.execute("SELECT util, n FROM samples_hourly WHERE hour=?",
                     (hour,)).fetchone()
    assert r["n"] == 2
    assert r["util"] == 60.0
    assert conn.execute("SELECT COUNT(*) c FROM samples").fetchone()["c"] == 0
