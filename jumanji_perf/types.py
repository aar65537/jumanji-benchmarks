import json
from dataclasses import dataclass
from glob import glob
from itertools import product
from math import isnan
from typing import Any, Dict, List


@dataclass
class CompileParams:
    batch_size: int
    double_wrap: bool


@dataclass
class RunParams:
    seed: int
    total_steps: int
    batch_size: int
    double_wrap: bool


@dataclass
class Benchmark:
    commit_hash: str
    platform: str
    name: str
    params: Dict[str, Any]
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
                        param = CompileParams(int(param[0]), bool(param[1]))
                    elif name == "time_rollout_run":
                        param = RunParams(
                            int(param[0]),
                            int(param[1]),
                            int(param[2]),
                            param[3] == "True",
                        )
                    else:
                        continue

                    if result is None or (isinstance(result, float) and isnan(result)):
                        continue

                    benchmarks.append(cls(commit_hash, platform, name, param, result))

        return benchmarks
