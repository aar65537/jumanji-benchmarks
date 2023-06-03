from typing import Tuple

import plotly.express as px
import plotly.graph_objects as go

from jumanji_perf.types import Benchmark, CompileParams, RunParams

BENCHMARKS = Benchmark.load()
COMMITS = list({benchmark.commit_hash for benchmark in BENCHMARKS})
PLATFORMS = sorted(list({benchmark.platform for benchmark in BENCHMARKS}))
WRAPPERS = sorted(list({benchmark.params.wrapper for benchmark in BENCHMARKS}))


def figures_by_commit(platform: str, wrapper: str) -> Tuple[go.Figure, go.Figure]:
    run_batch_sizes = []
    run_commits = []
    run_rates = []

    compile_batch_sizes = []
    compile_commits = []
    compile_time = []

    for benchmark in BENCHMARKS:
        if benchmark.platform != platform or benchmark.params.wrapper != wrapper:
            continue
        elif isinstance(benchmark.params, RunParams):
            run_batch_sizes.append(benchmark.params.batch_size)
            run_commits.append(benchmark.commit_hash)
            run_rates.append(benchmark.params.total_steps / benchmark.result)
        elif isinstance(benchmark.params, CompileParams):
            compile_batch_sizes.append(benchmark.params.batch_size)
            compile_commits.append(benchmark.commit_hash)
            compile_time.append(benchmark.result)

    run_fig = px.line(x=run_batch_sizes, y=run_rates, color=run_commits, markers=True)

    compile_fig = px.line(
        x=compile_batch_sizes, y=compile_time, color=compile_commits, markers=True
    )

    run_fig.update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Step Rate (hz)",
        legend_title="Commit",
    )

    compile_fig.update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Compile Time (sec)",
        legend_title="Commit",
    )

    return run_fig, compile_fig


def figures_by_platform(commit_hash: str, wrapper: str) -> Tuple[go.Figure, go.Figure]:
    run_batch_sizes = []
    run_platforms = []
    run_rates = []

    compile_batch_sizes = []
    compile_platforms = []
    compile_time = []

    for benchmark in BENCHMARKS:
        if benchmark.commit_hash != commit_hash or benchmark.params.wrapper != wrapper:
            continue
        elif isinstance(benchmark.params, RunParams):
            run_batch_sizes.append(benchmark.params.batch_size)
            run_platforms.append(benchmark.platform)
            run_rates.append(benchmark.params.total_steps / benchmark.result)
        elif isinstance(benchmark.params, CompileParams):
            compile_batch_sizes.append(benchmark.params.batch_size)
            compile_platforms.append(benchmark.platform)
            compile_time.append(benchmark.result)

    run_fig = px.line(x=run_batch_sizes, y=run_rates, color=run_platforms, markers=True)

    compile_fig = px.line(
        x=compile_batch_sizes, y=compile_time, color=compile_platforms, markers=True
    )

    run_fig.update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Step Rate (hz)",
        legend_title="Platform",
    )

    compile_fig.update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Compile Time (sec)",
        legend_title="Platform",
    )

    return run_fig, compile_fig


def figures_by_wrappers(commit_hash: str, platform: str) -> Tuple[go.Figure, go.Figure]:
    run_batch_sizes = []
    run_wrappers = []
    run_rates = []

    compile_batch_sizes = []
    compile_wrappers = []
    compile_time = []

    for benchmark in BENCHMARKS:
        if benchmark.commit_hash != commit_hash or benchmark.platform != platform:
            continue
        elif isinstance(benchmark.params, RunParams):
            run_batch_sizes.append(benchmark.params.batch_size)
            run_wrappers.append(benchmark.params.wrapper)
            run_rates.append(benchmark.params.total_steps / benchmark.result)
        elif isinstance(benchmark.params, CompileParams):
            compile_batch_sizes.append(benchmark.params.batch_size)
            compile_wrappers.append(benchmark.params.wrapper)
            compile_time.append(benchmark.result)

    run_fig = px.line(x=run_batch_sizes, y=run_rates, color=run_wrappers, markers=True)

    compile_fig = px.line(
        x=compile_batch_sizes, y=compile_time, color=compile_wrappers, markers=True
    )

    run_fig.update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Step Rate (hz)",
        legend_title="Wrapper",
    )

    compile_fig.update_layout(
        xaxis_title="Batch Size",
        yaxis_title="Compile Time (sec)",
        legend_title="Wrapper",
    )

    return run_fig, compile_fig
