import json
from dataclasses import dataclass
from glob import glob
from itertools import product
from math import isnan
from typing import List

from jumanji_benchmarks.settings import COMMITS


@dataclass
class Benchmark:
    commit: str
    platform: str
    seed: int
    total_steps: int
    batch_size: int
    wrapper: str
    result: float

    @classmethod
    def load(cls) -> List["Benchmark"]:
        benchmarks = []
        for result_file in glob("results/*/*py*.json"):
            with open(result_file, "r") as rf:
                results = json.load(rf)

            commit_hash = results["commit_hash"][:8]
            commit = COMMITS.get(commit_hash, commit_hash)
            platform = results["env_vars"]["JAX_PLATFORM_NAME"]
            benchmark = results["results"]["benchmarks.RunRolloutBenchmark.time_run"]

            params = product(*benchmark[1])
            results = benchmark[0]

            for param, result in zip(params, results):
                batch_size = int(param[1])
                if batch_size == 0:
                    batch_size = 1
                    wrapper = "AutoReset"
                elif param[2] == "True":
                    wrapper = "Vmap(AutoReset)"
                else:
                    wrapper = "VmapAutoReset"

                if platform == "cpu":
                    total_steps = 10**6
                else:
                    total_steps = 100 * batch_size

                if result is None or (isinstance(result, float) and isnan(result)):
                    continue

                benchmarks.append(
                    cls(
                        commit=commit,
                        platform=platform,
                        seed=int(param[0]),
                        total_steps=total_steps,
                        batch_size=batch_size,
                        wrapper=wrapper,
                        result=result,
                    )
                )

        return benchmarks
