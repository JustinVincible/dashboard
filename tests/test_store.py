import collector
import store
from collections import namedtuple

Snap = namedtuple("Snap", "cluster ok jobs finished remote_now")


def test_sacct_submit_and_start_are_stored(conn):
    snap = Snap("gaivi", True, [], [[
        "12345", "train", "COMPLETED", "01:00:00", "0:0",
        "2026-09-01T12:00:00", "8G",
        "2026-09-01T10:30:00", "2026-09-01T11:00:00"]], 0)
    store.write_snapshot(conn, snap)

    r = conn.execute("SELECT * FROM finished WHERE jid='12345'").fetchone()
    assert r["start_ts"] - r["submit_ts"] == 1800
    assert r["end_ts"] - r["start_ts"] == 3600
    assert r["first_written"] > 0


def test_short_sacct_rows_still_parse(conn):
    """An older sacct format, or a truncated line, must not drop the row."""
    snap = Snap("gaivi", True, [], [[
        "999", "old", "FAILED", "00:10:00", "1:0", "Unknown", ""]], 0)
    store.write_snapshot(conn, snap)

    r = conn.execute("SELECT * FROM finished WHERE jid='999'").fetchone()
    assert r["state"] == "FAILED"
    assert r["end_ts"] is None and r["submit_ts"] is None
    assert r["first_written"] > 0


def test_first_written_is_not_bumped_on_update(conn):
    snap = Snap("gaivi", True, [], [["1", "n", "RUNNING", "", "", "Unknown", ""]], 0)
    store.write_snapshot(conn, snap)
    first = conn.execute("SELECT first_written w FROM finished").fetchone()["w"]

    snap2 = Snap("gaivi", True, [], [["1", "n", "COMPLETED", "", "", "Unknown", ""]], 100000)
    store.write_snapshot(conn, snap2)
    r = conn.execute("SELECT * FROM finished").fetchone()
    assert r["first_written"] == first and r["state"] == "COMPLETED"


def test_migration_dates_rows_written_before_the_column_existed(conn, now):
    """A pre-migration row with no first_written must not fall out of every
    windowed query. It is dated by its end time, or by the migration moment."""
    conn.execute("INSERT INTO finished (cluster,jid,state,end_ts,first_written) "
                 "VALUES ('gaivi','dated','COMPLETED',?,NULL)", (now - 86400,))
    conn.execute("INSERT INTO finished (cluster,jid,state,end_ts,first_written) "
                 "VALUES ('gaivi','undated','FAILED',NULL,NULL)")
    conn.commit()

    store.init(conn)   # migration is idempotent and runs on every startup

    rows = {r["jid"]: r["first_written"]
            for r in conn.execute("SELECT jid, first_written FROM finished")}
    assert rows["dated"] == now - 86400
    assert rows["undated"] is not None


def test_piped_job_name_does_not_shift_fields():
    """sacct -P splits on "|" and JobName is user text, so a piped name must
    not push Submit and Start out of position."""
    f = collector.split_sacct(
        "469815|a|b|COMPLETED|00:10:00|0:0|2026-09-01T12:00:00|1G"
        "|2026-09-01T11:00:00|2026-09-01T11:50:00")
    assert f == ["469815", "a|b", "COMPLETED", "00:10:00", "0:0",
                 "2026-09-01T12:00:00", "1G",
                 "2026-09-01T11:00:00", "2026-09-01T11:50:00"]


def test_short_sacct_row_is_left_alone():
    f = collector.split_sacct("469815|run|COMPLETED|00:10:00|0:0|Unknown|1G")
    assert f[1] == "run" and len(f) == 7


# ------------------------------------------------------- cluster + memory ----

Snap2 = namedtuple("Snap2", "cluster ok jobs finished remote_now cluster_stats mem")

STATS = {"ALL": {"running": 22, "pending": 2, "other": 0, "nodes_total": 24,
                 "nodes_alloc": 1, "nodes_idle": 14, "nodes_mixed": 8,
                 "nodes_down": 1, "cpus_alloc": 322, "cpus_idle": 1752,
                 "cpus_total": 2200, "gpus_total": 99, "gpus_alloc": 41,
                 "gpu_models": {"A100": 15, "A40": 8}}}


def test_the_narrow_snapshot_fake_still_writes(conn):
    """write_snapshot reads the new fields off Snapshots that predate them."""
    store.write_snapshot(conn, Snap("gaivi", True, [], [], 1000))
    assert conn.execute("SELECT COUNT(*) c FROM cluster_samples").fetchone()["c"] == 0


def test_cluster_stats_are_written_per_partition(conn):
    store.write_snapshot(conn, Snap2("gaivi", True, [], [], 1000, STATS, {}))

    r = conn.execute("SELECT * FROM cluster_samples WHERE partition='ALL'").fetchone()
    assert (r["ts"], r["running"], r["gpus_alloc"], r["n"]) == (1000, 22, 41, 1)
    assert '"A100": 15' in r["gpu_models"]


def test_req_mem_lands_on_the_row_sacct_wrote(conn):
    """The memory query is separate from the history query, so the merge has to
    happen after the finished row exists."""
    snap = Snap2("gaivi", True, [], [[
        "470338_1", "train", "COMPLETED", "01:00:00", "0:0",
        "2026-09-01T12:00:00", "", "2026-09-01T11:00:00", "2026-09-01T11:00:00"]],
        1000, {}, {"470338_1": ("160G", "11534344K")})
    store.write_snapshot(conn, snap)

    r = conn.execute("SELECT * FROM finished WHERE jid='470338_1'").fetchone()
    assert (r["req_mem"], r["max_rss"]) == ("160G", "11534344K")


def test_a_half_reported_job_does_not_erase_what_is_stored(conn):
    """MaxRSS arrives an hour after ReqMem; the later tick must not blank it."""
    fin = [["470338_1", "t", "RUNNING", "00:10:00", "0:0", "Unknown", "", "", ""]]
    store.write_snapshot(conn, Snap2("gaivi", True, [], fin, 1000, {},
                                     {"470338_1": ("160G", "11534344K")}))
    store.write_snapshot(conn, Snap2("gaivi", True, [], fin, 1020, {},
                                     {"470338_1": ("160G", "")}))

    r = conn.execute("SELECT * FROM finished WHERE jid='470338_1'").fetchone()
    assert r["max_rss"] == "11534344K"


def test_the_new_columns_migrate_onto_an_old_database(conn):
    """A database created before power tracking must gain the columns, and the
    migration must survive being run twice."""
    for t in ("samples", "samples_hourly", "finished"):
        for col in store.ADDED_COLUMNS.get(t, {}):
            have = {r["name"] for r in conn.execute(f"PRAGMA table_info({t})")}
            assert col in have
    store.init(conn)
    store.init(conn)


def test_cluster_rollup_averages_and_records_how_many(conn):
    """Queue depth is a level, not an event, so an hour folds to its mean."""
    hour = 1_700_000_000 // 3600 * 3600
    for i, running in enumerate((10, 20, 30)):
        store.write_snapshot(conn, Snap2(
            "gaivi", True, [], [], hour + i * 20,
            {"ALL": {**STATS["ALL"], "running": running}}, {}))

    store._rollup_clusters(conn, hour + 86400)
    conn.commit()

    rows = conn.execute("SELECT * FROM cluster_samples").fetchall()
    assert len(rows) == 1
    assert rows[0]["running"] == 20 and rows[0]["n"] == 3
    assert rows[0]["ts"] == hour


def test_prune_drops_old_cluster_samples(conn):
    store.write_snapshot(conn, Snap2("gaivi", True, [], [], 1000, STATS, {}))
    store.prune(conn, keep_days=1)
    assert conn.execute("SELECT COUNT(*) c FROM cluster_samples").fetchone()["c"] == 0


def test_a_rolled_up_mean_keeps_its_fraction(conn):
    """Integer columns divided by an integer count would truncate every mean
    downward, so the average is taken in floating point."""
    hour = 1_700_000_000 // 3600 * 3600
    for i, running in enumerate((10, 20, 31)):
        store.write_snapshot(conn, Snap2(
            "gaivi", True, [], [], hour + i * 20,
            {"ALL": {**STATS["ALL"], "running": running}}, {}))

    store._rollup_clusters(conn, hour + 86400)
    conn.commit()

    r = conn.execute("SELECT running FROM cluster_samples").fetchone()
    assert abs(r["running"] - 61 / 3) < 1e-6


def test_rolling_up_twice_does_not_inflate_the_sample_count(conn):
    """Rollup runs daily over a window it has already collapsed once."""
    hour = 1_700_000_000 // 3600 * 3600
    for i in range(3):
        store.write_snapshot(conn, Snap2("gaivi", True, [], [], hour + i * 20,
                                         STATS, {}))
    store._rollup_clusters(conn, hour + 86400)
    store._rollup_clusters(conn, hour + 86400)
    conn.commit()

    rows = conn.execute("SELECT * FROM cluster_samples").fetchall()
    assert len(rows) == 1
    assert rows[0]["n"] == 3 and rows[0]["running"] == 22
