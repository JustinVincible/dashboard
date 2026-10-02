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

# ------------------------------------------------------------ capacity_use --

def test_capacity_counts_jobs_on_a_scheduler_and_gpus_on_a_workstation(conn, now):
    """A gaivi slot is one job whatever it holds; a wrx80 slot is one card."""
    bucket = 300
    base = (now - 600) // bucket * bucket
    seed_samples(conn, "gaivi", "1", base, bucket // INTERVAL, gpu_n=4)
    seed_samples(conn, "gaivi", "2", base, bucket // INTERVAL, gpu_n=4)
    seed_samples(conn, "wrx80", "p1", base, bucket // INTERVAL, gpu_n=2)

    out = queries.capacity_use(conn, {"gaivi": (6, "job"), "wrx80": (2, "gpu")},
                               days=7, bucket_s=bucket)
    assert out["series"]["gaivi"]["used"][0] == 2.0   # two jobs, not eight gpus
    assert out["series"]["wrx80"]["used"][0] == 2.0   # two cards, not one pane


def test_capacity_denominator_is_the_whole_window_not_the_observed_span(conn, now):
    """Idle hosts write no rows, so coverage must not shrink the denominator."""
    seed_samples(conn, "gaivi", "1", now - 3600, 3600 // INTERVAL, gpu_n=1)

    out = queries.capacity_use(conn, {"gaivi": (6, "job")}, days=7, bucket_s=1800)
    cl = out["clusters"][0]
    assert cl["slot_hours_avail"] == 6 * 7 * 24
    assert abs(cl["slot_hours_used"] - 1.0) < 0.05      # one job for one hour
    assert abs(cl["pct"] - 100 / 1008) < 0.05


def test_capacity_total_is_slot_hour_weighted_not_an_average_of_percentages(conn, now):
    """gaivi's six slots must outweigh the workstation's two in the headline."""
    seed_samples(conn, "gaivi", "1", now - 3600, 3600 // INTERVAL, gpu_n=1)
    seed_samples(conn, "wrx80", "p1", now - 3600, 3600 // INTERVAL, gpu_n=2)

    out = queries.capacity_use(conn, {"gaivi": (6, "job"), "wrx80": (2, "gpu")},
                               days=7, bucket_s=1800)
    by = {c["cluster"]: c for c in out["clusters"]}
    assert out["total"]["slot_hours_avail"] == 8 * 7 * 24
    mean_of_pcts = (by["gaivi"]["pct"] + by["wrx80"]["pct"]) / 2
    assert out["total"]["pct"] < mean_of_pcts
    assert abs(out["total"]["pct"]
               - 100 * out["total"]["slot_hours_used"] / (8 * 168)) < 0.05


def test_capacity_ignores_pending_jobs(conn, now):
    """util IS NOT NULL is the running-proxy; a queued job holds nothing."""
    seed_samples(conn, "gaivi", "1", now - 3600, 3600 // INTERVAL, gpu_n=1)
    conn.execute("INSERT INTO samples (ts,cluster,jid,state,util,gpu_n) "
                 "VALUES (?,'gaivi','2','PD',NULL,NULL)", (now - 1800,))
    conn.commit()

    out = queries.capacity_use(conn, {"gaivi": (6, "job")}, days=7, bucket_s=1800)
    assert max(out["series"]["gaivi"]["used"]) == 1.0


def test_capacity_survives_the_rollup(conn, now):
    """Raw rows folded into hours keep their slot-hours, both units."""
    slots = {"gaivi": (6, "job"), "wrx80": (2, "gpu")}
    start = now - 2 * 86400
    seed_samples(conn, "gaivi", "1", start, 3600 // INTERVAL + 1, gpu_n=4)
    seed_samples(conn, "wrx80", "p1", start, 3600 // INTERVAL + 1, gpu_n=2)

    before = queries.capacity_use(conn, slots, days=7, bucket_s=1800)
    assert store.rollup(conn, keep_days=1) > 0
    after = queries.capacity_use(conn, slots, days=7, bucket_s=1800)

    by_before = {c["cluster"]: c["slot_hours_used"] for c in before["clusters"]}
    by_after = {c["cluster"]: c["slot_hours_used"] for c in after["clusters"]}
    assert abs(by_after["gaivi"] - by_before["gaivi"]) < 0.05   # one job, not four gpus
    assert abs(by_after["wrx80"] - by_before["wrx80"]) < 0.05   # two cards
    assert abs(by_after["wrx80"] - 2.0) < 0.1


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


# ------------------------------------------------------------------ energy --

def test_energy_integrates_watts_over_time(conn, now):
    """One hour at a steady 300 W is 0.3 kWh, and costs the rate."""
    n = 3600 // INTERVAL + 1
    seed_samples(conn, "gaivi", "e1", now - 3600, n, power_w=300)

    out = queries.energy(conn, days=1, price_per_kwh=0.20)
    row = next(r for r in out["jobs"] if r["jid"] == "e1")
    assert abs(row["kwh"] - 0.3) < 0.005
    assert abs(row["dollars"] - 0.06) < 0.005


def test_energy_keeps_measured_and_estimated_apart(conn, now):
    """TDP x util reads low on an idle card, so the two must not be summed."""
    n = 3600 // INTERVAL + 1
    seed_samples(conn, "gaivi", "e2", now - 3600, n, power_w=300, power_est_w=100)

    row = next(r for r in queries.energy(conn, days=1)["jobs"] if r["jid"] == "e2")
    assert abs(row["kwh"] - 0.3) < 0.005
    assert abs(row["est_kwh"] - 0.1) < 0.005


def test_energy_ignores_an_outage_gap(conn, now):
    """A day the service was down is not a day of billable draw."""
    seed_samples(conn, "gaivi", "e3", now - 86400, 2, power_w=1000)
    seed_samples(conn, "gaivi", "e3", now - 60, 2, power_w=1000)

    row = next(r for r in queries.energy(conn, days=2)["jobs"] if r["jid"] == "e3")
    assert row["kwh"] < 2 * queries.CAP_S * 1000 / 3600.0 / 1000.0


def test_energy_sums_raw_and_rolled_up(conn, now):
    hour = (now - 10 * 86400) // 3600 * 3600
    conn.execute("INSERT INTO samples_hourly (hour,cluster,jid,util,gpu_used,gpu_n,n,dt_s,"
                 "power_w) VALUES (?,'gaivi','e4',100,8192,1,180,3600,500)", (hour,))
    conn.commit()

    row = next(r for r in queries.energy(conn, days=30)["jobs"] if r["jid"] == "e4")
    assert abs(row["kwh"] - 0.5) < 0.005


def test_a_job_with_no_power_reading_is_left_out(conn, now):
    seed_samples(conn, "gaivi", "e5", now - 3600, 10)
    assert not [r for r in queries.energy(conn, days=1)["jobs"] if r["jid"] == "e5"]


# ------------------------------------------------------- memory efficiency --

def test_mem_units_parse():
    assert queries.parse_mem_mib("56G") == 56 * 1024
    assert abs(queries.parse_mem_mib("39452548K") - 38527.9) < 0.1
    assert queries.parse_mem_mib("512M") == 512
    assert queries.parse_mem_mib("4Gn") == 4096      # older Slurm per-node suffix
    assert queries.parse_mem_mib("") is None
    assert queries.parse_mem_mib("0") is None


def test_mem_efficiency_ranks_the_worst_fit_first(conn, now):
    for jid, req, rss in (("m1", "56G", "39452548K"), ("m2", "160G", "8388608K")):
        conn.execute(
            "INSERT INTO finished (cluster,jid,name,state,end_ts,first_written,"
            "req_mem,max_rss) VALUES ('gaivi',?,?, 'COMPLETED',?,?,?,?)",
            (jid, jid, now - 600, now - 600, req, rss))
    conn.commit()

    rows = queries.mem_efficiency(conn, days=7)
    assert [r["jid"] for r in rows] == ["m2", "m1"]
    assert abs(rows[1]["pct"] - 67.2) < 0.5


def test_a_job_predating_the_mem_columns_is_skipped(conn, now):
    conn.execute("INSERT INTO finished (cluster,jid,state,end_ts,first_written) "
                 "VALUES ('gaivi','old','COMPLETED',?,?)", (now - 600, now - 600))
    conn.commit()
    assert not [r for r in queries.mem_efficiency(conn, days=7) if r["jid"] == "old"]


# ------------------------------------------------------------ cluster_stats --

def _seed_cluster(conn, ts, partition="ALL", running=10, pending=2, gpus_alloc=40):
    conn.execute(
        "INSERT OR REPLACE INTO cluster_samples (ts,cluster,partition,running,pending,"
        "other,nodes_total,nodes_alloc,nodes_idle,nodes_mixed,nodes_down,cpus_alloc,"
        "cpus_idle,cpus_total,gpus_total,gpus_alloc,gpu_models,n) "
        "VALUES (?,'gaivi',?,?,?,0,24,1,14,8,1,322,1752,2200,99,?,'{\"A100\": 15}',1)",
        (ts, partition, running, pending, gpus_alloc))
    conn.commit()


def test_cluster_stats_splits_the_total_from_the_partitions(conn, now):
    _seed_cluster(conn, now - 40)
    _seed_cluster(conn, now - 40, partition="general", running=9, gpus_alloc=14)

    d = queries.cluster_stats(conn, days=1)["gaivi"]
    assert d["total"]["running"] == 10
    assert [p["partition"] for p in d["partitions"]] == ["general"]
    assert d["total"]["gpu_models"] == {"A100": 15}


def test_cluster_stats_reports_only_the_newest_row(conn, now):
    _seed_cluster(conn, now - 600, running=1)
    _seed_cluster(conn, now - 20, running=7)
    assert queries.cluster_stats(conn, days=1)["gaivi"]["total"]["running"] == 7


def test_the_load_series_averages_without_truncating(conn, now):
    """Three samples in one bucket average to 20.33, and integer division would
    render that as a flat 20."""
    base = (now - 60) // 300 * 300 + 10          # inside one bucket
    for i, running in enumerate((10, 20, 31)):
        _seed_cluster(conn, base + i, running=running)

    s = queries.cluster_stats(conn, days=1, bucket_s=300)["gaivi"]["series"]
    assert s["running"] == [20.3]


def test_live_draw_is_the_newest_reading_not_an_arbitrary_one(conn, now):
    """The panel's watts must match what the card reports now, so an older
    sample in the same window must not win."""
    for i, w in enumerate((900.0, 292.3)):
        conn.execute(
            "INSERT INTO samples (ts,cluster,jid,state,gpu_n,power_w) "
            "VALUES (?,'gaivi','470338_2','R',1,?)", (now - 60 + i * 40, w))
    conn.commit()

    jobs = queries.energy(conn, days=1)["jobs"]
    assert [j["watts"] for j in jobs] == [292.3]
