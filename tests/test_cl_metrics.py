"""M10 -- continual-learning metrics (criteria G2, G3)."""

import pytest

from amaun.errors import EvaluationError
from amaun.evaluation.cl_metrics import PerformanceMatrix, compute


def _known_matrix():
    """A hand-computable 3-task matrix.

    S = [[0.90, 0.15,  -  ],
         [0.80, 0.70, 0.20],
         [0.60, 0.65, 0.85]]
    AvgPerf  = (0.60 + 0.65 + 0.85)/3                       = 0.700
    Forget   = mean(max(0, 0.90-0.60), max(0, 0.70-0.65))   = mean(0.30, 0.05) = 0.175
    BWT      = mean(0.60-0.90, 0.65-0.70)                   = mean(-0.30, -0.05) = -0.175
    FWT      = mean(S[t1][t2]-b, S[t2][t3]-b), b = 0.10     = mean(0.05, 0.10) = 0.075
    """
    matrix = PerformanceMatrix(
        task_order=["t1", "t2", "t3"],
        zero_shot_baseline={"t1": 0.10, "t2": 0.10, "t3": 0.10},
    )
    matrix.record("t1", "t1", 0.90)
    matrix.record("t1", "t2", 0.15)
    matrix.record("t2", "t1", 0.80)
    matrix.record("t2", "t2", 0.70)
    matrix.record("t2", "t3", 0.20)
    matrix.record("t3", "t1", 0.60)
    matrix.record("t3", "t2", 0.65)
    matrix.record("t3", "t3", 0.85)
    return matrix


def test_all_four_metrics_match_hand_computation():
    metrics = compute(_known_matrix())
    assert metrics.average_performance == pytest.approx(0.700, abs=1e-9)
    assert metrics.forgetting == pytest.approx(0.175, abs=1e-9)
    assert metrics.backward_transfer == pytest.approx(-0.175, abs=1e-9)
    assert metrics.forward_transfer == pytest.approx(0.075, abs=1e-9)
    assert metrics.n_tasks == 3


def test_forgetting_is_clipped_but_bwt_is_signed():
    """idea.md S4.5 clips forgetting; BWT is signed. Both are reported so neither hides."""
    matrix = PerformanceMatrix(task_order=["t1", "t2"], zero_shot_baseline={"t1": 0.0, "t2": 0.0})
    matrix.record("t1", "t1", 0.50)
    matrix.record("t2", "t1", 0.80)     # IMPROVED on the old task
    matrix.record("t2", "t2", 0.60)
    metrics = compute(matrix)
    assert metrics.forgetting == pytest.approx(0.0)       # clipped at zero
    assert metrics.backward_transfer == pytest.approx(0.30)   # positive, visible


def test_forward_transfer_is_undefined_without_a_baseline():
    """A-06: FWT needs b_j; substituting zero would silently invent a value."""
    matrix = PerformanceMatrix(task_order=["t1", "t2"])
    matrix.record("t1", "t1", 0.5)
    matrix.record("t1", "t2", 0.2)
    matrix.record("t2", "t1", 0.4)
    matrix.record("t2", "t2", 0.6)
    metrics = compute(matrix)
    assert metrics.forward_transfer is None
    assert any("forward transfer" in note for note in metrics.notes)


def test_partial_evaluation_is_flagged_not_silently_averaged():
    matrix = PerformanceMatrix(task_order=["t1", "t2", "t3"])
    matrix.record("t3", "t1", 0.5)
    matrix.record("t3", "t3", 0.7)     # t2 missing
    metrics = compute(matrix)
    assert metrics.average_performance == pytest.approx(0.6)
    assert any("2/3" in note for note in metrics.notes)


def test_empty_matrix_is_an_error():
    with pytest.raises(EvaluationError):
        compute(PerformanceMatrix(task_order=[]))
    with pytest.raises(EvaluationError):
        compute(PerformanceMatrix(task_order=["t1"]))


def test_single_task_matrix_has_no_forgetting_or_transfer():
    matrix = PerformanceMatrix(task_order=["t1"])
    matrix.record("t1", "t1", 0.8)
    metrics = compute(matrix)
    assert metrics.average_performance == pytest.approx(0.8)
    assert metrics.forgetting is None
    assert metrics.backward_transfer is None


def test_matrix_persists_and_metrics_recompute_identically(tmp_path):
    """Criterion G3: metrics must be recomputable without re-running the experiment."""
    original = _known_matrix()
    path = original.save(tmp_path / "matrix.json")
    reloaded = PerformanceMatrix.load(path)
    assert reloaded.task_order == original.task_order
    assert reloaded.zero_shot_baseline == original.zero_shot_baseline
    assert compute(reloaded).as_dict() == compute(original).as_dict()


def test_grid_view_shape():
    grid = _known_matrix().as_grid()
    assert len(grid) == 3 and all(len(row) == 3 for row in grid)
    assert grid[0][2] is None       # t3 not yet seen after t1
