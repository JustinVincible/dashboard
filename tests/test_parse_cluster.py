"""Cluster-wide queue and node state.

The sinfo and squeue blocks here are verbatim capture from gaivi, so a change
in either command's output shape breaks these rather than the capacity panel.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collector  # noqa: E402

QUEUE = """      1 PENDING|Quick,Contributors
      1 PENDING|general
     12 RUNNING|Contributors
      2 RUNNING|Extended
      9 RUNNING|general""".splitlines()

# Partition StateLong Nodes CPUsState Gres GresUsed. One row per
# partition+state pair, so GPU46 shows up under general, Quick and Contributors.
SINFO = """general*      mixed      1  16/80/0/96    gpu:A100:3     gpu:A100:2(IDX:0,2)
general*      idle       5  0/160/0/160   gpu:1080Ti:4   gpu:1080Ti:0(IDX:N/A)
general*      allocated  1  40/0/0/40     gpu:1080Ti:8   gpu:1080Ti:8(IDX:0-7)
Quick         mixed      1  16/80/0/96    gpu:A100:3     gpu:A100:2(IDX:0,2)
Contributors  mixed      1  16/80/0/96    gpu:A100:3     gpu:A100:2(IDX:0,2)
Contributors  draining   1  2/0/126/128   gpu:RTXPRO6000BW:8  gpu:RTXPRO6000BW:0(IDX:N/A)
Contributors  mixed      1  20/44/0/64    gpu:A100:2,gpu:L40:2  gpu:A100:0(IDX:N/A),gpu:L40:2(IDX:2-3)
""".splitlines()

# NodeList StateLong CPUsState Gres GresUsed, deduped to one row per node.
NODES = """GPU22  allocated  40/0/0/40    gpu:1080Ti:8   gpu:1080Ti:8(IDX:0-7)
GPU46  mixed      16/80/0/96   gpu:A100:3     gpu:A100:2(IDX:0,2)
GPU49  mixed      20/44/0/64   gpu:A100:2,gpu:L40:2  gpu:A100:0(IDX:N/A),gpu:L40:2(IDX:2-3)
GPU57  draining   2/0/126/128  gpu:RTXPRO6000BW:8  gpu:RTXPRO6000BW:0(IDX:N/A)
GPU3   idle       0/32/0/32    gpu:1080Ti:4   gpu:1080Ti:0(IDX:N/A)
""".splitlines()


def parsed():
    return collector.parse_cluster(QUEUE, SINFO, NODES)


def test_the_default_partition_loses_its_star():
    p = parsed()
    assert "general" in p and "general*" not in p


def test_repeated_partition_rows_fold_into_one():
    """sinfo emits a row per partition+state, so general appears three times."""
    g = parsed()["general"]
    assert g["nodes_total"] == 7          # 1 mixed + 5 idle + 1 allocated
    assert g["nodes_idle"] == 5
    assert g["nodes_mixed"] == 1
    assert g["nodes_alloc"] == 1


def test_the_cpu_state_field_splits_into_alloc_idle_total():
    g = parsed()["general"]
    assert g["cpus_alloc"] == 16 + 0 + 40
    assert g["cpus_idle"] == 80 + 5 * 160 + 0
    assert g["cpus_total"] == 96 + 5 * 160 + 40


def test_gres_counts_scale_by_the_node_count():
    """`gpu:1080Ti:4` on 5 idle nodes is 20 cards, not 4."""
    g = parsed()["general"]
    assert g["gpu_models"]["1080Ti"] == 4 * 5 + 8
    assert g["gpus_alloc"] == 2 + 0 + 8


def test_a_node_with_two_gpu_types_counts_both():
    c = parsed()["Contributors"]
    assert c["gpu_models"]["L40"] == 2
    assert c["gpu_models"]["A100"] == 3 + 2


def test_the_total_comes_from_the_node_view_not_the_partition_sum():
    """GPU46 sits in three partitions. Summing them would triple it."""
    p = parsed()
    assert p["ALL"]["nodes_total"] == 5              # five distinct nodes
    assert p["ALL"]["gpus_total"] == 8 + 3 + 2 + 2 + 8 + 4
    assert p["ALL"]["gpus_alloc"] == 8 + 2 + 2
    assert p["ALL"]["nodes_total"] < sum(
        v["nodes_total"] for k, v in p.items() if k != "ALL")


def test_a_draining_node_is_neither_idle_nor_busy():
    assert parsed()["ALL"]["nodes_down"] == 1


def test_a_pending_job_counts_toward_every_partition_it_named():
    p = parsed()
    assert p["Quick"]["pending"] == 1
    assert p["Contributors"]["pending"] == 1
    assert p["ALL"]["pending"] == 2       # two pending jobs, counted once each


def test_running_counts_land_on_the_right_partitions():
    p = parsed()
    assert p["general"]["running"] == 9
    assert p["Extended"]["running"] == 2
    assert p["ALL"]["running"] == 23


def test_empty_output_is_an_empty_cluster_not_a_crash():
    p = collector.parse_cluster([], [], [])
    assert p["ALL"]["running"] == 0 and p["ALL"]["nodes_total"] == 0


def test_a_node_with_no_gpus_reports_none():
    p = collector.parse_cluster([], [], ["cpu1  idle  0/64/0/64  (null)  (null)"])
    assert p["ALL"]["gpus_total"] == 0 and p["ALL"]["nodes_idle"] == 1


# ----------------------------------------------------------- sacct memory ----

MEM = """470338_1|160G|
470338_1.batch||11534344K
470340|16G|
470343_1|160G|
470343_1.batch||
""".splitlines()


def test_req_mem_and_max_rss_join_on_the_base_job_id():
    """ReqMem is on the allocation line, MaxRSS on the .batch step."""
    m = collector.parse_mem(MEM)
    assert m["470338_1"] == ("160G", "11534344K")


def test_a_job_still_running_keeps_the_half_it_has():
    m = collector.parse_mem(MEM)
    assert m["470343_1"] == ("160G", "")
    assert m["470340"] == ("16G", "")
