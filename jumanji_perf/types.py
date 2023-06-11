import json
from dataclasses import dataclass
from glob import glob
from itertools import product
from math import isnan
from typing import List

COMMITS = {
    "a1ae4440": "0: main",
    "2e9f0186": "1: switch inside move",
    "ca2e4ba5": "2: no mutate in cond",
    "8f9a67bd": "3: vmap over move_left_row",
    "e431a5e9": "4: use fori_loop",
    "77fccd6e": "5: single pass move",
    "88a51285": "6: implement can move",
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
