import chex
import jax
import pytest

from jumanji.environments.logic.game_2048.types import State
from jumanji_perf.runners import RolloutRunner


@pytest.fixture
def key() -> chex.PRNGKey:
    return jax.random.PRNGKey(0)


def test_rollout_runner__rollout(key: chex.PRNGKey) -> None:
    runner = RolloutRunner(key, 5, 0, False)
    state = runner.rollout()
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == tuple()
    assert state.step_count == 5

    runner.total_steps = 10
    state = runner.rollout()
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == tuple()
    assert state.step_count == 10

    runner.batch_size = 2
    state = runner.rollout()
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == (2,)
    assert state.step_count[0] == 5

    runner.total_steps = 20
    state = runner.rollout()
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == (2,)
    assert state.step_count[0] == 10


def test_rollout_runner__rollout_jit(key: chex.PRNGKey) -> None:
    rollout = jax.jit(chex.assert_max_traces(RolloutRunner.rollout, n=2))

    runner = RolloutRunner(key, 5, 0, False)
    state = rollout(runner)
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == tuple()
    assert state.step_count == 5

    runner.total_steps = 10
    state = rollout(runner)
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == tuple()
    assert state.step_count == 10

    runner.batch_size = 2
    state = rollout(runner)
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == (2,)
    assert state.step_count[0] == 5

    runner.total_steps = 20
    state = rollout(runner)
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == (2,)
    assert state.step_count[0] == 10

    runner.batch_size = 0
    state = rollout(runner)
    assert isinstance(state, State)
    assert isinstance(state.step_count, chex.Array)
    assert state.step_count.shape == tuple()
    assert state.step_count == 20

    runner.batch_size = 4
    with pytest.raises(AssertionError):
        state = rollout(runner)
