from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


import adaptive_campaign_metrics as metrics  # noqa: E402
import fold_mpnn_esmfold2 as folding  # noqa: E402


def test_filter_threshold_operators_have_requested_boundary_semantics() -> None:
    assert not metrics.criterion(0.5, "gt", 0.5)["pass"]
    assert not metrics.criterion(6, "lt", 6)["pass"]
    assert metrics.criterion(2, "ge", 2)["pass"]
    assert metrics.criterion(5, "ge", 5)["pass"]
    assert not metrics.criterion(None, "gt", 0.5)["pass"]


def test_alignment_and_directional_rmsd_primitives() -> None:
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = target + np.asarray([10.0, -4.0, 2.0])
    assert metrics.aligned_rmsd(mobile, target) < 1e-10
    assert metrics.rmsd(mobile, target) > 1.0


def test_fold_state_parser_and_task_seed_are_stable() -> None:
    assert folding.parse_states("AB_SEP,DC_SER") == {"AB_SEP", "DC_SER"}
    assert folding.parse_states(None) is None
    assert folding.stable_task_seed(123, "design_AB_SEP") == folding.stable_task_seed(
        123, "design_AB_SEP"
    )
    assert folding.stable_task_seed(123, "design_AB_SEP") != folding.stable_task_seed(
        123, "design_DC_SER"
    )
