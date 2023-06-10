import plotly.express as px
import plotly.graph_objects as go

from jumanji_perf.types import Benchmark

BENCHMARKS = Benchmark.load()
COMMITS = list({benchmark.commit for benchmark in BENCHMARKS})
DEFAULT_COMMIT = COMMITS[0] if len(COMMITS) else None
PLATFORMS = sorted(list({benchmark.platform for benchmark in BENCHMARKS}))
DEFAULT_PLATFORM = PLATFORMS[0] if len(PLATFORMS) else None
WRAPPERS = sorted(list({benchmark.wrapper for benchmark in BENCHMARKS}))
DEFAULT_WRAPPER = WRAPPERS[0] if len(WRAPPERS) else None
OPTIONS = {"log_x": True, "markers": True}


def figures_by_commit(platform: str, wrapper: str) -> go.Figure:
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


def figures_by_platform(commit: str, wrapper: str) -> go.Figure:
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


def figures_by_wrappers(commit_hash: str, platform: str) -> go.Figure:
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
