"""ParallelKrigingExecutor must return what the model's own predict() returns.

Each test here failed before the executor was fixed: thread and process results
were joined in the order tasks finished, fitted models could not be sent to worker
processes, the spatial strategy left targets on a line without a tile, worker
errors became NaN, and progress_callback was skipped when tqdm was installed.
"""

import time

import numpy as np
import pytest
from pygeostats.kriging import OrdinaryKriging, ParallelKrigingExecutor, spatial_tiles
from pygeostats.variogram import EmpiricalVariogram, Variogram


@pytest.fixture(scope="module")
def kriging():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, size=(60, 2))
    values = np.sin(coords[:, 0] / 2) + np.cos(coords[:, 1] / 3)
    values += rng.normal(0, 0.1, len(coords))
    empirical = EmpiricalVariogram(coords, values, n_bins=10).compute()
    variogram = Variogram(model="exponential").fit(
        empirical.distances_, empirical.gamma_, weights=empirical.counts_
    )
    return OrdinaryKriging(variogram).fit(coords, values)


@pytest.fixture(scope="module")
def targets():
    # A shuffled grid, so that predictions put in the wrong rows cannot match.
    xs = np.linspace(0, 10, 12)
    grid = np.column_stack([axis.ravel() for axis in np.meshgrid(xs, xs)])
    return grid[np.random.default_rng(0).permutation(len(grid))]


class _LaterChunksFinishFirst:
    """Predicts each target's x coordinate, sleeping longer for earlier chunks.

    With one worker per chunk, the chunks finish in reverse order, which exposes
    results joined in completion order rather than submission order.
    """

    def predict(self, coordinates, return_variance=False):
        coordinates = np.asarray(coordinates)
        chunk_index = int(coordinates[0, 0] // 10)
        time.sleep(0.03 * (3 - chunk_index))
        predictions = coordinates[:, 0].copy()
        if return_variance:
            return predictions, np.zeros(len(coordinates))
        return predictions


class _FailsBeyondX:
    """Raises for any chunk containing a target with x above a threshold."""

    def __init__(self, threshold):
        self.threshold = threshold

    def predict(self, coordinates, return_variance=False):
        coordinates = np.asarray(coordinates)
        if np.any(coordinates[:, 0] > self.threshold):
            raise ValueError("prediction failed")
        predictions = coordinates[:, 0].copy()
        if return_variance:
            return predictions, np.zeros(len(coordinates))
        return predictions


@pytest.mark.parametrize("strategy", ["chunk", "spatial", "adaptive"])
@pytest.mark.parametrize("method", ["sequential", "thread"])
def test_predictions_match_predict(kriging, targets, method, strategy):
    expected, expected_variance = kriging.predict(targets, return_variance=True)
    executor = ParallelKrigingExecutor(
        n_workers=4, execution_method=method, chunk_size=10
    )

    predictions, variance = executor.predict_parallel(
        kriging, targets, return_variance=True, strategy=strategy
    )
    np.testing.assert_allclose(predictions, expected)
    np.testing.assert_allclose(variance, expected_variance)

    only_predictions = executor.predict_parallel(kriging, targets, strategy=strategy)
    np.testing.assert_allclose(only_predictions, expected)


@pytest.mark.parametrize("strategy", ["chunk", "spatial"])
def test_process_workers_match_predict(kriging, targets, strategy):
    expected, expected_variance = kriging.predict(targets, return_variance=True)
    executor = ParallelKrigingExecutor(
        n_workers=2, execution_method="process", chunk_size=50
    )

    predictions, variance = executor.predict_parallel(
        kriging, targets, return_variance=True, strategy=strategy
    )
    np.testing.assert_allclose(predictions, expected)
    np.testing.assert_allclose(variance, expected_variance)


def test_thread_chunks_are_joined_in_target_order():
    targets = np.column_stack([np.arange(40.0), np.zeros(40)])
    executor = ParallelKrigingExecutor(
        n_workers=4, execution_method="thread", chunk_size=10
    )

    predictions = executor.predict_parallel(_LaterChunksFinishFirst(), targets)

    np.testing.assert_array_equal(predictions, targets[:, 0])


@pytest.mark.parametrize(
    "layout", ["horizontal line", "vertical line", "single point", "repeated point"]
)
@pytest.mark.parametrize("method", ["sequential", "thread"])
def test_spatial_strategy_predicts_every_target(kriging, method, layout):
    line = np.linspace(0, 10, 30)
    targets = {
        "horizontal line": np.column_stack([line, np.full(line.size, 5.0)]),
        "vertical line": np.column_stack([np.full(line.size, 5.0), line]),
        "single point": np.array([[3.0, 4.0]]),
        "repeated point": np.tile([3.0, 4.0], (25, 1)),
    }[layout]
    executor = ParallelKrigingExecutor(
        n_workers=2, execution_method=method, chunk_size=5
    )

    predictions = executor.predict_parallel(kriging, targets, strategy="spatial")

    np.testing.assert_allclose(predictions, kriging.predict(targets))


@pytest.mark.parametrize("threshold", [-1.0, 8.0], ids=["every task", "some tasks"])
@pytest.mark.parametrize("strategy", ["chunk", "spatial"])
@pytest.mark.parametrize("method", ["sequential", "thread"])
def test_worker_errors_are_raised(targets, method, strategy, threshold):
    executor = ParallelKrigingExecutor(
        n_workers=2, execution_method=method, chunk_size=20
    )

    with pytest.raises(ValueError, match="prediction failed"):
        executor.predict_parallel(_FailsBeyondX(threshold), targets, strategy=strategy)


@pytest.mark.parametrize("strategy", ["chunk", "spatial"])
@pytest.mark.parametrize("method", ["sequential", "thread"])
def test_progress_callback_reports_every_task(kriging, targets, method, strategy):
    calls = []
    executor = ParallelKrigingExecutor(
        n_workers=2,
        execution_method=method,
        chunk_size=20,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    executor.predict_parallel(kriging, targets, strategy=strategy)

    assert calls
    total = calls[-1][1]
    assert [done for done, _ in calls] == list(range(1, total + 1))
    assert {reported_total for _, reported_total in calls} == {total}


def test_spatial_tiles_cover_bounds_with_no_width():
    tiles = spatial_tiles((0.0, 5.0, 10.0, 5.0), 2.5, overlap=0.0)

    assert tiles
    assert min(tile[0] for tile in tiles) == 0.0
    assert max(tile[2] for tile in tiles) == 10.0
    assert all(tile[1] == 5.0 and tile[3] == 5.0 for tile in tiles)


@pytest.mark.parametrize("tile_size", [0.0, -1.0, (1.0, 0.0)])
def test_spatial_tiles_reject_sizes_that_are_not_positive(tile_size):
    with pytest.raises(ValueError, match="tile_size"):
        spatial_tiles((0.0, 0.0, 10.0, 10.0), tile_size)
