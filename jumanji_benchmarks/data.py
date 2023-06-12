from jumanji_benchmarks.types import Benchmark

BENCHMARKS = Benchmark.load()
BATCH_SIZES = sorted(list({benchmark.batch_size for benchmark in BENCHMARKS}))
DEFAULT_BATCH_SIZE = BATCH_SIZES[0] if len(BATCH_SIZES) else None
COMMITS = sorted(list({benchmark.commit for benchmark in BENCHMARKS}))
DEFAULT_COMMIT = COMMITS[0] if len(COMMITS) else None
PLATFORMS = sorted(list({benchmark.platform for benchmark in BENCHMARKS}))
DEFAULT_PLATFORM = PLATFORMS[0] if len(PLATFORMS) else None
WRAPPERS = sorted(list({benchmark.wrapper for benchmark in BENCHMARKS}))
DEFAULT_WRAPPER = WRAPPERS[0] if len(WRAPPERS) else None
