import json
from dataclasses import dataclass
from glob import glob
from itertools import product
from math import isnan
from typing import List

COMMITS = {
    "d22bb170": "main",
    "77d66fde": "remove switch",
    "d09eff0a": "improve move",
    "c253a4d7": "implement can move",
    "bf17fb12": "improve move + remove switch",
    "04d7e714": "improve move + remove switch (rot coord)",
    "c68f8cc7": "improve move + remove switch (rot coord 2)",
    "11874c0a": "improve move + remove switch (rot coord 3)",
    "c3614934": "remove switch 2",
    "02c8bec2": "remove switch 3",
    "f3daf2d3": "improve-move-2",
    "5079b725": "improve-move take 2",
    "75f4b0fc": "improve-move take 3",
    "96ab9d86": "vmap over cols",
    "32ec5aec": "vmap over rows",
}


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
            benchmark = results["results"]["time_rollout_run"]

            params = product(*benchmark[1])
            results = benchmark[0]

            for param, result in zip(params, results):
                batch_size = int(param[2])
                if batch_size == 0:
                    batch_size = 1
                    wrapper = "AutoReset"
                elif param[3] == "True":
                    wrapper = "Vmap(AutoReset)"
                else:
                    wrapper = "VmapAutoReset"

                if result is None or (isinstance(result, float) and isnan(result)):
                    continue

                benchmarks.append(
                    cls(
                        commit=commit,
                        platform=platform,
                        seed=int(param[0]),
                        total_steps=int(param[1]),
                        batch_size=batch_size,
                        wrapper=wrapper,
                        result=result,
                    )
                )

        return benchmarks
