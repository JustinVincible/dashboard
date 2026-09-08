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
