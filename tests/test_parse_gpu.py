"""nvidia-smi field mapping.

The rows here are verbatim captures. A widened parser breaks here first, which
is the point -- a silently dropped row reads on the dashboard as "gpu: n/a".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collector  # noqa: E402


def test_six_field_row_carries_power_and_model():
    raw = "40960, 81920, 99, 276.51, 400.00, NVIDIA A100-SXM4-80GB;"
    stats, err = collector.parse_gpu(raw)
    used, total, util, n, watts, est, name = stats
    assert (used, total, util, n) == (40960, 81920, 99, 1)
    assert watts == 276.5
    assert est == 396.0                    # 400 W TDP x 99%
    assert name == "NVIDIA A100-SXM4-80GB"
    assert err == ""


def test_several_cards_sum_memory_and_power_and_mean_util():
    raw = ("20000, 46068, 100, 300.00, 300.00, NVIDIA A40;"
           "20000, 46068, 98, 280.00, 300.00, NVIDIA A40;")
    stats, _ = collector.parse_gpu(raw)
    assert stats[0] == 40000 and stats[3] == 2
    assert stats[2] == 99                  # mean, not sum
    assert stats[4] == 580.0


def test_a_card_that_will_not_report_power_falls_back_to_tdp():
    raw = "1000, 24564, 50, [N/A], [N/A], NVIDIA GeForce GTX 1080 Ti;"
    stats, _ = collector.parse_gpu(raw)
    assert stats[4] is None                # nothing measured
    assert stats[5] == 125.0               # 250 W TDP x 50%


def test_a_partial_power_reading_is_dropped_not_halved():
    """One card reporting and one not would understate the pair's draw."""
    raw = ("20000, 46068, 100, 300.00, 300.00, NVIDIA A40;"
           "20000, 46068, 100, [N/A], 300.00, NVIDIA A40;")
    stats, _ = collector.parse_gpu(raw)
    assert stats[3] == 2 and stats[4] is None


def test_the_old_three_field_shape_still_parses():
    """Captures predating power.draw must not vanish from the job table."""
    stats, err = collector.parse_gpu("8192, 40960, 75;")
    assert stats[:4] == (8192, 40960, 75, 1)
    assert stats[4] is None and err == ""


def test_an_srun_refusal_comes_back_as_a_reason_not_a_reading():
    stats, err = collector.parse_gpu(
        "srun: Job step creation temporarily disabled, retrying;")
    assert stats is None
    assert err.startswith("Job step creation")


def test_nothing_at_all_is_neither_a_reading_nor_an_error():
    assert collector.parse_gpu("") == (None, "")


def test_tdp_matches_the_longest_model_key():
    assert collector.tdp_for("NVIDIA A100-SXM4-80GB") == 400
    assert collector.tdp_for("NVIDIA A100-PCIE-40GB") == 250
    assert collector.tdp_for("NVIDIA RTX PRO 6000 Blackwell Server Edition") == 600
    assert collector.tdp_for("Some Unlisted Card") is None
