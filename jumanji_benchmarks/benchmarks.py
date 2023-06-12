import os
from typing import Any

import jax

from jumanji_benchmarks.runners import RolloutRunner

SEED = [0]
BATCH_SIZES = [0, 1, 10, 10**2, 10**3, 10**4, 10**5, 10**6]
DOUBLE_WRAPS = [False, True]


class RunRolloutBenchmark:
    params = (SEED, BATCH_SIZES, DOUBLE_WRAPS)
    param_names = ["seed", "batch_size", "double_wrap"]
    timeout = 120

    def setup(self, seed: int, batch_size: int, double_wrap: bool) -> None:
        key = jax.random.PRNGKey(seed)

        if os.environ.get("JAX_PLATFORM_NAME") == "cpu":
            total_steps = 10**6
        elif os.environ.get("JAX_PLATFORM_NAME") == "cuda":
            if batch_size > 10**4 and not double_wrap:
                raise NotImplementedError
            total_steps = 100 * (1 if batch_size == 0 else batch_size)

        runner = RolloutRunner(key, total_steps, batch_size, double_wrap)
        self.rollout = jax.jit(runner.rollout)
        self.rollout.lower().compile()

    def time_run(self, *_: Any) -> None:
        jax.block_until_ready(self.rollout())
