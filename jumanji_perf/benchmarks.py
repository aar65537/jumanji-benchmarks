import os
from typing import Any

import jax

from jumanji_perf.runners import RolloutRunner

SEED = [0]
TOTAL_STEPS = [10**2, 10**5, 10**8]
BATCH_SIZES = [0, 1, 10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
DOUBLE_WRAPS = [False, True]


class CompileRolloutBenchmark:
    params = (BATCH_SIZES, DOUBLE_WRAPS)
    param_names = ["batch_size", "double_wrap"]

    def setup(self, batch_size: int, double_wrap: bool) -> None:
        key = jax.random.PRNGKey(0)
        runner = RolloutRunner(key, batch_size + 1, batch_size, double_wrap)

        self.rollout = jax.jit(runner.rollout)

    def time_compile(self, *_: Any) -> None:
        self.rollout.lower().compile()

    time_compile.benchmark_name = "time_rollout_compile"  # type: ignore


class RunRolloutBenchmark:
    params = (SEED, TOTAL_STEPS, BATCH_SIZES, DOUBLE_WRAPS)
    param_names = ["seed", "total_steps", "batch_size", "double_wrap"]

    def setup(
        self, seed: int, total_steps: int, batch_size: int, double_wrap: bool
    ) -> None:
        key = jax.random.PRNGKey(seed)
        runner = RolloutRunner(key, total_steps, batch_size, double_wrap)

        if os.environ.get("JAX_PLATFORM_NAME") == "cpu" and total_steps != 10**5:
            raise NotImplementedError("Only benchmark 100,000 total steps on cpu.")
        elif os.environ.get("JAX_PLATFORM_NAME") == "cuda" and runner.n_steps > 100:
            raise NotImplementedError("Don't benchmark more than 100 steps on gpu.")

        self.rollout = jax.jit(runner.rollout)
        self.rollout.lower().compile()

    def time_run(self, *_: Any) -> None:
        jax.block_until_ready(self.rollout())

    time_run.benchmark_name = "time_rollout_run"  # type: ignore
