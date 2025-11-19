
#include "xstl/xstl.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <vector>

static void BM_BuildBinaryAssociatorCPU(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto nvalues = state.range(0);
    std::vector<std::int32_t> values(nvalues);
    std::ranges::copy(std::views::iota(0, nvalues), values.data());
    std::vector<std::int32_t> keys(nvalues);
    std::ranges::transform(
        std::views::iota(0, nvalues), keys.data(), [](auto x) -> int32_t { return x % 2 == 0; });
    state.ResumeTiming();

    xstd::association_map<std::int32_t> associator(nvalues, 2u);
    associator.fill(keys, values);
  }
}

BENCHMARK(BM_BuildBinaryAssociatorCPU)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_MAIN();
