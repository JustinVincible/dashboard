"""Background refresh loop.

The only thing in the process that talks to the clusters. It owns the SSH
cadence, writes every tick to SQLite, and publishes the newest snapshot in
memory. HTTP handlers read that and never block on ssh, so page loads stay
instant no matter how slow gaivi is feeling.
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import store
from alerts import alerts_for
from collector import Settings, collect, load_config, minutes_left


def job_dict(snap, j) -> dict:
    d = asdict(j)
    d["name"] = j.name.strip()
    d["partition"] = j.partition.strip()
    d["minutes_left"] = minutes_left(j.timeleft)
    d["log_age_min"] = round((snap.remote_now - j.mtime) / 60, 1) if j.mtime else None
    d["series"] = j.series[-120:]
    return d


class Poller:
    def __init__(self, st: Settings | None = None, conn=None):
        self.st = st or load_config()
        self.conn = conn or store.connect()
        store.init(self.conn)
        self.latest: dict = {"ts": 0, "clusters": [], "jobs": [], "alerts": [], "ok": False}
        self.generation = 0
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None
        self.last_error = ""
        self._last_maint = 0.0

    # ---------------------------------------------------------------- loop --

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def refresh_now(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.tick()
            except Exception:
                self.last_error = traceback.format_exc(limit=3)
                print(self.last_error, flush=True)
            self._wake.wait(max(2.0, self.st.interval - (time.time() - t0)))
            self._wake.clear()

    # ---------------------------------------------------------------- tick --

    def tick(self) -> None:
        clusters = self.st.clusters
        with ThreadPoolExecutor(max_workers=max(1, len(clusters))) as ex:
            snaps = list(ex.map(
                lambda c: collect(c, self.st, True, True, self.st.history_days), clusters))

        active = []
        for s in snaps:
            store.write_snapshot(self.conn, s)
            if s.ok:
                active += alerts_for(s, self.st)

        fresh = store.sync_alerts(self.conn, active)
        self._maintain()
        if self.st.notify and fresh:
            self._notify(fresh)

        payload = {
            "ts": int(time.time()),
            "ok": any(s.ok for s in snaps),
            "clusters": [{"name": s.cluster, "ok": s.ok, "error": s.error, "warn": s.warn,
                          "remote_now": s.remote_now,
                          "running": sum(1 for j in s.jobs if j.state == "R"),
                          "pending": sum(1 for j in s.jobs if j.state == "PD"),
                          "total": len(s.jobs)} for s in snaps],
            "jobs": [job_dict(s, j) for s in snaps if s.ok for j in s.jobs],
            "alerts": self.active_alerts(),
        }
        with self._cv:
            self.latest = payload
            self.generation += 1
            self._cv.notify_all()

    def _maintain(self) -> None:
        """Roll up and prune once a day. This runs for months under KeepAlive, so
        housekeeping cannot live at startup only."""
        if time.time() - self._last_maint < 86400:
            return
        self._last_maint = time.time()
        try:
            folded = store.rollup(self.conn)
            store.prune(self.conn)
            if folded:
                print(f"rollup: folded {folded} sample rows into hours", flush=True)
        except Exception:
            self.last_error = traceback.format_exc(limit=3)
            print(self.last_error, flush=True)

    def active_alerts(self) -> list:
        rows = self.conn.execute(
            "SELECT key,kind,cluster,jid,msg,first_seen,last_seen,acked FROM alerts "
            "WHERE cleared_at IS NULL ORDER BY first_seen DESC")
        return [dict(r) for r in rows]

    def _notify(self, fresh) -> None:
        body = "; ".join(m for _, m in fresh[:3])
        self._notify_mac(body)
        if self.st.notify_url:
            # Off the poller thread. A slow or unreachable webhook must not
            # stretch the ssh cadence.
            threading.Thread(target=self._notify_web, args=(body,),
                             name="notify", daemon=True).start()

    def _notify_mac(self, body: str) -> None:
        import subprocess
        import sys
        if sys.platform != "darwin":
            return
        subprocess.run(["osascript", "-e",
                        f'display notification "{body.replace(chr(34), chr(39))}" '
                        f'with title "jobdash"'], capture_output=True)

    def _notify_web(self, body: str) -> None:
        """POST to ntfy, Slack, or Discord. The Mac only sees notifications while
        someone is looking at it, and this runs headless for weeks."""
        import json as _json
        import urllib.error
        import urllib.request

        url = self.st.notify_url
        if any(h in url for h in ("slack.com", "discord.com", "discordapp.com")):
            key = "content" if "discord" in url else "text"
            data = _json.dumps({key: f"jobdash: {body}"}).encode()
            headers = {"Content-Type": "application/json"}
        else:
            data = body.encode()
            headers = {"Title": "jobdash", "Content-Type": "text/plain"}
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            urllib.request.urlopen(req, timeout=10).read()
        except (urllib.error.URLError, OSError) as e:
            print(f"notify_url failed: {e}", flush=True)

    # --------------------------------------------------------------- watch --

    def wait_for_change(self, seen: int, timeout: float = 25.0) -> int:
        """Block until a tick newer than `seen` lands. Backs the SSE stream."""
        with self._cv:
            if self.generation > seen:
                return self.generation
            self._cv.wait(timeout)
            return self.generation


if __name__ == "__main__":
    p = Poller()
    p.tick()
    print(f"{len(p.latest['jobs'])} jobs, {len(p.latest['alerts'])} active alerts")
    for c in p.latest["clusters"]:
        print(" ", c["name"], "ok" if c["ok"] else f"FAIL {c['error']}",
              c["running"], "running /", c["total"], "total")
