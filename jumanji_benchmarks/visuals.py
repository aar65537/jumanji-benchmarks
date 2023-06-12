from typing import Dict, List, Tuple

import plotly.express as px
import plotly.graph_objects as go

from jumanji_benchmarks.data import BENCHMARKS
from jumanji_benchmarks.settings import OPTIONS


def table_data(batch_size: int, platform: str) -> List[Dict[str, str]]:
    entries: List[Tuple[str, float]] = []

    for benchmark in BENCHMARKS:
        if benchmark.batch_size != batch_size or benchmark.platform != platform:
            continue
        if benchmark.batch_size == 1:
            if benchmark.wrapper != "AutoReset":
                continue
        else:
            if benchmark.platform == "cpu" and benchmark.wrapper != "VmapAutoReset":
                continue
            if benchmark.platform == "cuda" and benchmark.wrapper != "Vmap(AutoReset)":
                continue
        entry = (benchmark.commit, benchmark.total_steps / benchmark.result)
        entries.append(entry)

    data = [
        {
            "commit": commit,
            "rate": f"{rate:.3e}",
            "iterative": f"{rate / entries[max(i - 1, 0)][1] - 1:.3%}",
            "total": f"{rate / entries[0][1] - 1 :.3%}",
        }
        for i, (commit, rate) in enumerate(entries)
    ]

    return data


def figure_by_commit(platform: str, wrapper: str) -> go.Figure:
    batch_sizes = []
    commits = []
    rates = []

    for benchmark in BENCHMARKS:
        if benchmark.platform != platform or benchmark.wrapper != wrapper:
            continue
        batch_sizes.append(benchmark.batch_size)
        commits.append(benchmark.commit)
        rates.append(benchmark.total_steps / benchmark.result)

    return px.line(x=batch_sizes, y=rates, color=commits, **OPTIONS).update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Step Rate (hz)",
        legend_title="Commit",
    )


def figure_by_platform(commit: str, wrapper: str) -> go.Figure:
    batch_sizes = []
    platforms = []
    rates = []

    for benchmark in BENCHMARKS:
        if benchmark.commit != commit or benchmark.wrapper != wrapper:
            continue
        batch_sizes.append(benchmark.batch_size)
        platforms.append(benchmark.platform)
        rates.append(benchmark.total_steps / benchmark.result)

    return px.line(x=batch_sizes, y=rates, color=platforms, **OPTIONS).update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Step Rate (hz)",
        legend_title="Platform",
    )


def figure_by_wrappers(commit_hash: str, platform: str) -> go.Figure:
    batch_sizes = []
    wrappers = []
    rates = []

    for benchmark in BENCHMARKS:
        if benchmark.commit != commit_hash or benchmark.platform != platform:
            continue
        batch_sizes.append(benchmark.batch_size)
        wrappers.append(benchmark.wrapper)
        rates.append(benchmark.total_steps / benchmark.result)

    return px.line(x=batch_sizes, y=rates, color=wrappers, **OPTIONS).update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Step Rate (hz)",
        legend_title="Wrapper",
    )
