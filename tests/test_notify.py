"""Webhook shaping. The transport is stubbed — what matters is that a Slack URL
gets JSON, an ntfy topic gets plain text, and neither can raise into the tick."""
import urllib.error

import poller
from collector import Settings


class FakePoller(poller.Poller):
    def __init__(self, url):
        self.st = Settings(notify_url=url)


def capture(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["data"] = req.data
        seen["headers"] = req.headers

        class R:
            def read(self):
                return b""
        return R()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def test_slack_webhook_gets_json(monkeypatch):
    seen = capture(monkeypatch)
    FakePoller("https://hooks.slack.com/services/x")._notify_web("job 1 stalled")
    assert seen["data"] == b'{"text": "jobdash: job 1 stalled"}'
    assert seen["headers"]["Content-type"] == "application/json"


def test_ntfy_topic_gets_plain_text(monkeypatch):
    seen = capture(monkeypatch)
    FakePoller("https://ntfy.sh/mytopic")._notify_web("job 1 stalled")
    assert seen["data"] == b"job 1 stalled"
    assert seen["headers"]["Title"] == "jobdash"


def test_an_unreachable_webhook_does_not_raise(monkeypatch, capsys):
    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    FakePoller("https://ntfy.sh/mytopic")._notify_web("anything")
    assert "notify_url failed" in capsys.readouterr().out
