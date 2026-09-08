"""The job id in /api/logtail reaches a remote shell, so the guard on it matters
more than the happy path."""
import collector
from collector import Cluster


def test_a_non_numeric_job_id_is_refused_before_any_ssh(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("ssh must not run for a rejected job id")

    monkeypatch.setattr(collector, "ssh", boom)
    for bad in ["1; rm -rf ~", "$(whoami)", "../x", "abc", "", "`id`"]:
        path, text = collector.log_tail(Cluster("c", "h", "u"), bad)
        assert path == "" and "refusing" in text


def test_a_real_job_id_reaches_ssh_and_the_file_line_is_stripped(monkeypatch):
    sent = {}

    def fake_ssh(host, cmd, timeout=None, opts=None):
        sent["cmd"] = cmd
        return collector.SSHResult(True, out="@@FILE /home/u/run_3.log\nline one\nline two")

    monkeypatch.setattr(collector, "ssh", fake_ssh)
    path, text = collector.log_tail(Cluster("c", "h", "u"), "469815_1", lines=50)
    assert path == "/home/u/run_3.log"
    assert text == "line one\nline two"
    assert "tail -n 50" in sent["cmd"]


def test_line_count_is_clamped(monkeypatch):
    sent = {}

    def fake_ssh(host, cmd, timeout=None, opts=None):
        sent["cmd"] = cmd
        return collector.SSHResult(True, out="")

    monkeypatch.setattr(collector, "ssh", fake_ssh)
    collector.log_tail(Cluster("c", "h", "u"), "1", lines=99999)
    assert "tail -n 500" in sent["cmd"]


def test_a_failed_ssh_reports_the_reason(monkeypatch):
    monkeypatch.setattr(collector, "ssh",
                        lambda *a, **kw: collector.SSHResult(False, reason="timed out"))
    path, text = collector.log_tail(Cluster("c", "h", "u"), "1")
    assert path == "" and text == "(timed out)"
