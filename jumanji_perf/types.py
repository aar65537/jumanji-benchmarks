import json
from dataclasses import dataclass
from glob import glob
from itertools import product
from math import isnan
from typing import List, Union


@dataclass
class CompileParams:
    batch_size: int
    wrapper: str


@dataclass
class RunParams:
    seed: int
    total_steps: int
    batch_size: int
    wrapper: str


@dataclass
class Benchmark:
    commit_hash: str
    platform: str
    name: str
    params: Union[CompileParams, RunParams]
    result: float

    @classmethod
    def load(cls) -> List["Benchmark"]:
        benchmarks = []
        for result_file in glob("results/*/*py*.json"):
            with open(result_file, "r") as rf:
                results = json.load(rf)

            commit_hash = results["commit_hash"][:8]
            platform = results["env_vars"]["JAX_PLATFORM_NAME"]

            for name, benchmark in results["results"].items():
                params = product(*benchmark[1])
                results = benchmark[0]

                for param, result in zip(params, results):
                    if name == "time_rollout_compile":
                        batch_size = int(param[0])
                        if batch_size == 0:
                            batch_size = 1
                            wrapper = "AutoReset"
                        elif param[1] == "True":
                            wrapper = "Vmap(AutoReset)"
                        else:
                            wrapper = "VmapAutoReset"

                        param = CompileParams(batch_size, wrapper)

                    elif name == "time_rollout_run":
                        batch_size = int(param[2])
                        if batch_size == 0:
                            batch_size = 1
                            wrapper = "AutoReset"
                        elif param[3] == "True":
                            wrapper = "Vmap(AutoReset)"
                        else:
                            wrapper = "VmapAutoReset"

                        param = RunParams(
                            int(param[0]), int(param[1]), batch_size, wrapper
                        )
                    else:
                        continue

                    if result is None or (isinstance(result, float) and isnan(result)):
                        continue

                    benchmarks.append(cls(commit_hash, platform, name, param, result))

        return benchmarks
