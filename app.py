#!/usr/bin/env python3
"""jobdash web service.

Serves the dashboard on 127.0.0.1. Every route reads SQLite or the poller's
in-memory snapshot, so page loads never wait on ssh. `/api/logtail` is the one
exception, and it is opt-in per click — it runs its ssh on a worker thread with
the cluster's own timeout, so a slow cluster delays that one request and nothing
else. Read-only by design — there is no route that can touch a job on the
cluster.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import collector
import queries
import store
from poller import Poller

HERE = Path(__file__).parent
app = FastAPI(title="jobdash")
poller: Poller | None = None


@app.on_event("startup")
def _startup():
    global poller
    poller = Poller()
    store.backfill_jsonl(poller.conn, Path.home() / ".local/state/jobdash/history.jsonl")
    store.rollup(poller.conn)
    store.prune(poller.conn)
    poller.start()


def q(fn, *args, **kw):
    """Run a read against the poller's connection, serialized with its writes.

    sqlite3 cannot take concurrent statements on one connection, and the poller
    writes on this same handle from its own thread. Without this, any endpoint
    can fail with "bad parameter or other API misuse" mid-refresh.
    """
    with store._lock:
        return fn(poller.conn, *args, **kw)


@app.get("/")
def index():
    return FileResponse(HERE / "static/index.html")


@app.get("/api/live")
def live():
    return JSONResponse(poller.latest)


@app.get("/api/config")
def config():
    st = poller.st
    return {"interval": st.interval, "stall_minutes": st.stall_minutes,
            "low_util_pct": st.low_util_pct, "timeleft_warn_min": st.timeleft_warn_min,
            "price_per_kwh": st.price_per_kwh,
            "clusters": [{"name": c.name, "kind": c.kind, "gpu": c.gpu,
                          "slots": c.slots, "slot_unit": collector.slot_unit(c)}
                         for c in st.clusters]}


@app.get("/api/series/{cluster}/{jid}")
def series(cluster: str, jid: str, days: float = 7):
    return q(queries.job_series, cluster, jid, days)


@app.get("/api/logtail/{cluster}/{jid}")
async def logtail(cluster: str, jid: str, lines: int = 80):
    """The last N lines of a running job's log, fetched when asked for."""
    cl = next((c for c in poller.st.clusters if c.name == cluster), None)
    if cl is None:
        return JSONResponse({"path": "", "text": f"(unknown cluster {cluster})"})
    path, text = await asyncio.to_thread(collector.log_tail, cl, jid, lines)
    return {"path": path, "text": text}


@app.get("/api/squeue")
async def squeue():
    """Full squeue per slurm cluster, fetched on demand. Clusters whose last
    poll failed are skipped rather than left to time out."""
    up = {c["name"] for c in poller.latest.get("clusters", []) if c["ok"]}
    slurm = [c for c in poller.st.clusters if c.kind == "slurm"]
    live = [c for c in slurm if c.name in up]
    texts = await asyncio.gather(*(asyncio.to_thread(collector.squeue_all, c) for c in live))
    out = dict(zip((c.name for c in live), texts))
    return [{"name": c.name, "ok": c.name in out,
             "text": out.get(c.name, "(not connected)")} for c in slurm]


@app.get("/api/cluster_util")
def cluster_util(days: float = 1, bucket: int = 300):
    return q(queries.cluster_util, days, bucket)


@app.get("/api/capacity")
def capacity(days: float = 7, bucket: int = 1800):
    slots = {c.name: (c.slots, collector.slot_unit(c))
             for c in poller.st.clusters if c.slots > 0}
    return q(queries.capacity_use, slots, days, bucket)


@app.get("/api/gpu_hours")
def gpu_hours(days: float = 30):
    return q(queries.gpu_hours, days)


@app.get("/api/clusters")
def clusters(days: float = 1, bucket: int = 300):
    return q(queries.cluster_stats, days, bucket)


@app.get("/api/energy")
def energy(days: float = 7):
    return q(queries.energy, days, poller.st.price_per_kwh)


@app.get("/api/mem_efficiency")
def mem_efficiency(days: float = 30):
    return q(queries.mem_efficiency, days)


@app.get("/api/queue_waits")
def queue_waits(days: float = 30):
    return q(queries.queue_waits, days)


@app.get("/api/failures")
def failures(days: float = 30):
    return q(queries.failures, days)


@app.get("/api/finished")
def finished(days: float = 7):
    return q(queries.recent_finished, days)


@app.get("/api/timeline")
def timeline(days: float = 7):
    return q(queries.timeline, days)


@app.get("/api/compare")
def compare(job: list[str] = Query(default=[]), days: float = 30):
    """?job=cluster/jid repeated. Loss curves for a sweep, on one step axis."""
    pairs = [tuple(j.split("/", 1)) for j in job if "/" in j][:12]
    return q(queries.compare_loss, pairs, days)


@app.get("/api/alerts")
def alerts(days: float = 14):
    return q(queries.alert_history, days)


@app.post("/api/alerts/{key:path}/ack")
def ack(key: str):
    store.ack_alert(poller.conn, key)
    poller.latest["alerts"] = poller.active_alerts()
    return {"ok": True}


@app.post("/api/refresh")
def refresh():
    poller.refresh_now()
    return {"ok": True}


@app.get("/events")
async def events(request: Request):
    """SSE. One message per poller tick, so the browser mirrors the ssh cadence
    instead of polling on its own clock."""
    async def gen():
        seen = -1
        while True:
            if await request.is_disconnected():
                return
            gen_id = await asyncio.to_thread(poller.wait_for_change, seen, 25.0)
            if gen_id > seen:
                seen = gen_id
                yield f"data: {json.dumps(poller.latest)}\n\n"
            else:
                yield ": keepalive\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class NoCacheStatic(StaticFiles):
    """Revalidate every asset. This is a localhost app served off an SSD, so a
    stale cached app.js after an edit costs far more than the round-trip does.
    """

    def file_response(self, *a, **kw):
        r = super().file_response(*a, **kw)
        r.headers["Cache-Control"] = "no-cache"
        return r


app.mount("/vendor", StaticFiles(directory=HERE / "vendor"), name="vendor")
app.mount("/static", NoCacheStatic(directory=HERE / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()

    # Fail on a missing config here rather than inside uvicorn's startup, where
    # the one useful line arrives at the bottom of a traceback.
    if not collector.CONFIG_PATH.exists():
        sys.exit(f"no config at {collector.CONFIG_PATH} — run "
                 f"`collector.py --write-config` and edit the [[cluster]] blocks in it")

    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
