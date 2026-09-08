"""In-memory fixtures. Every aggregation test seeds samples at a known cadence
so the expected GPU-hours and utilization are arithmetic, not eyeballed."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import store  # noqa: E402

INTERVAL = 20   # seconds between samples, matching the shipped default


@pytest.fixture
def conn():
    c = store.connect(":memory:")
    store.init(c)
    yield c
    c.close()


@pytest.fixture
def now():
    return int(time.time())


def seed_samples(conn, cluster, jid, start_ts, n, util=50, gpu_n=2,
                 gpu_used=8192, step=INTERVAL):
    for i in range(n):
        conn.execute(
            "INSERT INTO samples (ts,cluster,jid,state,gpu_used,gpu_total,util,gpu_n) "
            "VALUES (?,?,?,'R',?,?,?,?)",
            (start_ts + i * step, cluster, jid, gpu_used, 40960, util, gpu_n))
    conn.commit()
