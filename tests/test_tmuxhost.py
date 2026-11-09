"""The tmux backend's field mapping and its synthesized job endings.

The remote stdout here is verbatim capture from the workstation, so a change in
the script's output shape breaks these rather than the dashboard.
"""
from collections import namedtuple

import store
import tmuxhost
from collector import FS, Cluster, Settings

Snap = namedtuple("Snap", "cluster ok jobs finished remote_now")
Job = namedtuple("Job", "jid")

CL = Cluster("wrx80", "wrx80", "justin", kind="tmux")
ST = Settings()

GPU_UUID = "GPU-7d83f849-a96e-20a4-dbc7-f54224e44989"

# btime 1788446546 + 51600513 ticks / 100 Hz = 1788962551
LOGFILE_JOB = FS.join([
    "jdtest", "0", "0", "train", "%0", "381778", "381779", "51600513", "521",
    "python3", "python3 -u train.py --lr 3e-4", "381779 381900 ",
    "/data/runs/train_07.log", "1788963003", "0", "0.91;0.44;0.30;",
    "step 1129 loss=0.3012",
])

PANE_JOB = FS.join([
    "jdgpu", "1", "0", "sweep-a", "%3", "382817", "382817", "51650000", "38",
    "python", "python -u sweep.py", "382817 382900 ", "", "1788963072", "2",
    "25.5;25.4;", "step 124 loss=25.4",
])


def out(*jobs, gpu=True, power=True):
    parts = ["@@NOW", "1788963003", "@@BOOT", "1788446546", "@@HZ", "100"]
    if gpu:
        tail = ", 214.83, NVIDIA GeForce RTX 3090" if power else ""
        parts += ["@@GPU",
                  f"0, GPU-1d40c244-b57b-7214-f4e2-f39b8f6e9a22, 15, 24564, 0"
                  + (", 21.4, NVIDIA GeForce RTX 3090" if power else ""),
                  f"1, {GPU_UUID}, 740, 24564, 60" + tail,
                  "@@APPS", f"382817, 740, {GPU_UUID}"]
    return "\n".join(parts + ["@@JOBS", *jobs])


def test_empty_workstation_is_ok_not_broken():
    """No tmux server is the normal state of an idle box, not a failure."""
    snap = tmuxhost.parse(CL, ST, out())
    assert snap.ok and snap.jobs == []


def test_logfile_job_maps_onto_the_slurm_shape():
    j = tmuxhost.parse(CL, ST, out(LOGFILE_JOB)).jobs[0]
    assert j.jid == "381779_1788962551"
    assert j.name == "train"           # the tmux window name wins
    assert j.state == "R"
    assert j.partition == "jdtest"
    assert j.node == "jdtest:0.0"      # no GPU, so no card suffix
    assert j.elapsed == "00:08:41"
    assert j.timeleft == "UNLIMITED"
    assert j.log == "train_07.log"
    assert j.mtime == 1788963003
    assert j.series == [0.91, 0.44, 0.30]
    assert j.lastline == "step 1129 loss=0.3012"


def test_gpu_is_attributed_through_the_process_tree():
    j = tmuxhost.parse(CL, ST, out(PANE_JOB)).jobs[0]
    used, total, util, n = j.gpu[:4]
    assert (used, total, util, n) == (740, 24564, 60, 1)
    assert j.gpu_err == ""
    assert j.node == "jdgpu:1.0 gpu1"     # names the card it holds


def test_power_is_read_off_the_card_the_job_holds():
    j = tmuxhost.parse(CL, ST, out(PANE_JOB)).jobs[0]
    assert j.gpu[4] == 214.8              # measured, not the idle card's 21.4
    assert "3090" in j.gpu[6]


def test_a_capture_without_power_still_parses():
    """The 5-field inventory predates power.draw; it must not drop the job."""
    j = tmuxhost.parse(CL, ST, out(PANE_JOB, power=False)).jobs[0]
    assert j.gpu[:4] == (740, 24564, 60, 1)
    assert j.gpu[4] is None


def test_a_job_holding_no_gpu_says_so():
    j = tmuxhost.parse(CL, ST, out(LOGFILE_JOB)).jobs[0]
    assert j.gpu is None and j.gpu_err == "cpu only"


def test_elapsed_parses_back_into_a_real_start_time():
    """`store.run_start` reads this, and a Gantt bar depends on it."""
    j = tmuxhost.parse(CL, ST, out(LOGFILE_JOB)).jobs[0]
    assert store.run_start(1788963003, j) == 1788963003 - 521


def test_jid_separates_two_runs_that_reuse_a_pid():
    """pid alone would concatenate both runs' loss curves under one job."""
    later = LOGFILE_JOB.replace(FS + "51600513" + FS, FS + "99900000" + FS)
    jids = {j.jid for j in tmuxhost.parse(CL, ST, out(LOGFILE_JOB, later)).jobs}
    assert jids == {"381779_1788962551", "381779_1789445546"}


def test_jid_survives_the_sacct_step_suffix_strip():
    """store.write_snapshot does jid.split('.')[0] on finished rows."""
    jid = tmuxhost.parse(CL, ST, out(LOGFILE_JOB)).jobs[0].jid
    assert "." not in jid and tmuxhost.JID_RE.match(jid)


def test_name_falls_back_to_the_script_not_the_interpreter():
    assert tmuxhost.job_name("bash", "python3", "python3 -u train.py --lr 1") == "train.py"
    assert tmuxhost.job_name("", "python3", "python3 -m torch.distributed") == "torch.distributed"
    assert tmuxhost.job_name("", "python3", "python3 -c import x") == "python3"
    assert tmuxhost.job_name("sweep-a", "python3", "python3 train.py") == "sweep-a"


def test_hms_is_a_format_minutes_left_understands():
    from collector import minutes_left
    assert tmuxhost.hms(521) == "00:08:41"
    assert tmuxhost.hms(90061) == "1-01:01:01"
    assert minutes_left(tmuxhost.hms(90061)) == 1501 + 1 / 60


def test_malformed_records_are_skipped_not_fatal():
    snap = tmuxhost.parse(CL, ST, out("garbage", LOGFILE_JOB, FS.join(["a", "b"])))
    assert len(snap.jobs) == 1


# --------------------------------------------------- synthesized endings --

def seed_running(conn, jid, run_start, last_seen, name="train"):
    conn.execute(
        "INSERT INTO jobs (cluster,jid,name,partition,node,first_seen,last_seen,"
        "run_start,last_state) VALUES ('wrx80',?,?,'s','s:0.0',?,?,?,'R')",
        (jid, name, run_start, last_seen, run_start))
    conn.commit()


def test_a_vanished_job_gets_an_end(conn):
    seed_running(conn, "1_100", 100, 1000)
    store.close_vanished(conn, Snap("wrx80", True, [], [], 2000))

    r = conn.execute("SELECT * FROM finished").fetchone()
    assert r["state"] == "ENDED"
    assert r["end_ts"] == 1000        # when we last saw it, not now
    assert r["start_ts"] == 100
    assert r["elapsed"] == "00:15:00"
    assert r["submit_ts"] is None     # never pended, so never a queue wait


def test_a_still_running_job_is_left_alone(conn):
    seed_running(conn, "1_100", 100, 1000)
    store.close_vanished(conn, Snap("wrx80", True, [Job("1_100")], [], 2000))
    assert conn.execute("SELECT COUNT(*) c FROM finished").fetchone()["c"] == 0


def test_a_failed_ssh_does_not_end_every_job(conn):
    seed_running(conn, "1_100", 100, 1000)
    store.close_vanished(conn, Snap("wrx80", False, [], [], 0))
    assert conn.execute("SELECT COUNT(*) c FROM finished").fetchone()["c"] == 0


def test_an_end_is_written_once(conn):
    """The jobs row keeps last_state='R' forever, so the guard is `finished`."""
    seed_running(conn, "1_100", 100, 1000)
    store.close_vanished(conn, Snap("wrx80", True, [], [], 2000))
    conn.execute("UPDATE finished SET name='edited'")
    store.close_vanished(conn, Snap("wrx80", True, [], [], 3000))

    r = conn.execute("SELECT * FROM finished").fetchone()
    assert r["name"] == "edited" and r["end_ts"] == 1000


def test_another_machines_jobs_are_untouched(conn):
    seed_running(conn, "1_100", 100, 1000)
    conn.execute("UPDATE jobs SET cluster='gaivi'")
    conn.commit()
    store.close_vanished(conn, Snap("wrx80", True, [], [], 2000))
    assert conn.execute("SELECT COUNT(*) c FROM finished").fetchone()["c"] == 0


# ------------------------------------------- a half-run script is not idle --

def test_a_script_that_died_partway_is_a_failure_not_an_idle_box():
    """An empty snapshot is how this backend says every job ended, so a broken
    script must not produce one."""
    snap = tmuxhost.parse(CL, ST, "@@NOW\n1788963003\n@@JOBS", err="/proc: denied")
    assert not snap.ok
    assert "died partway" in snap.error and "/proc: denied" in snap.error


def test_a_missing_boot_time_never_yields_a_pid_only_jid():
    """Without it every job would key on `pid_0` and two reruns would merge."""
    broken = "\n".join(["@@NOW", "1788963003", "@@BOOT", "0", "@@HZ", "100",
                        "@@JOBS", LOGFILE_JOB])
    assert not tmuxhost.parse(CL, ST, broken).ok


def test_remote_stderr_with_no_jobs_surfaces_as_a_warning():
    snap = tmuxhost.parse(CL, ST, out(), err="pgrep: command not found")
    assert snap.ok and snap.warn == "pgrep: command not found"
    assert tmuxhost.parse(CL, ST, out(LOGFILE_JOB), err="noise").warn == ""


def test_a_job_that_comes_back_has_its_end_row_removed(conn):
    """One bad snapshot must not leave a run permanently finished."""
    seed_running(conn, "1_100", 100, 1000)
    assert store.close_vanished(conn, Snap("wrx80", True, [], [], 2000)) == 1

    store.close_vanished(conn, Snap("wrx80", True, [Job("1_100")], [], 3000))
    assert conn.execute("SELECT COUNT(*) c FROM finished").fetchone()["c"] == 0


def test_reopening_leaves_other_finished_rows_alone(conn):
    seed_running(conn, "1_100", 100, 1000)
    seed_running(conn, "2_200", 100, 1000)
    store.close_vanished(conn, Snap("wrx80", True, [], [], 2000))

    store.close_vanished(conn, Snap("wrx80", True, [Job("1_100")], [], 3000))
    assert [r["jid"] for r in conn.execute("SELECT jid FROM finished")] == ["2_200"]


def test_a_real_end_is_never_reopened(conn):
    """Only ENDED rows are synthesized here, so only those may be undone."""
    conn.execute("INSERT INTO finished (cluster,jid,state,first_written) "
                 "VALUES ('wrx80','1_100','COMPLETED',1000)")
    conn.commit()
    store.close_vanished(conn, Snap("wrx80", True, [Job("1_100")], [], 2000))
    assert conn.execute("SELECT COUNT(*) c FROM finished").fetchone()["c"] == 1


def test_cpu_watts_from_rapl_pair():
    # 50 J over 0.5 s -> 100 W
    assert tmuxhost.parse_cpu_watts(["1000000 51000000 0 500000000 65532610987"]) == 100.0


def test_cpu_watts_counter_wrap():
    assert tmuxhost.parse_cpu_watts(["99000000 1000000 0 1000000000 100000000"]) == 2.0


def test_cpu_watts_unreadable():
    assert tmuxhost.parse_cpu_watts([]) is None


def test_script_finds_a_per_user_tmux_on_a_bare_ssh_path():
    script = tmuxhost.build_script("loss", 20, False, False)
    assert script.splitlines()[0] == 'PATH="$PATH:$HOME/.local/bin"'
