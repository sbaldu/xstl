
#include "xstl/xstl.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <hip_runtime.h>
#include <ranges>

static void BM_BuildBinaryAssociatorHIP(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto nvalues = state.range(0);
    auto values = xstd::hip::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::copy(std::views::iota(0) | std::views::take(nvalues), values.data());
    auto keys = xstd::hip::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::transform(std::views::iota(0) | std::views::take(nvalues),
                           keys.data(),
                           [](auto x) -> int32_t { return x % 2 == 0; });
    auto d_keys = xstd::hip::make_device_unique<std::int32_t[]>(nvalues);
    auto d_values = xstd::hip::make_device_unique<std::int32_t[]>(nvalues);
    hipMemcpy(d_keys.data(), keys.data(), nvalues * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    hipMemcpy(
        d_values.data(), values.data(), nvalues * sizeof(std::int32_t), hipMemcpyHostToDevice);
    state.ResumeTiming();

    xstd::hip::association_map<std::int32_t> associator(nvalues, 2u);
    associator.fill(d_keys, d_values);
  }
}

static void BM_BuildBinaryAssociatorHIPStreamed(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto nvalues = state.range(0);
    auto values = xstd::hip::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::copy(std::views::iota(0) | std::views::take(nvalues), values.data());
    auto keys = xstd::hip::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::transform(std::views::iota(0) | std::views::take(nvalues),
                           keys.data(),
                           [](auto x) -> int32_t { return x % 2 == 0; });

    hipStream_t stream;
    hipStreamCreate(&stream);
    auto d_keys = xstd::hip::make_device_unique<std::int32_t[]>(nvalues, stream);
    auto d_values = xstd::hip::make_device_unique<std::int32_t[]>(nvalues, stream);
    hipMemcpyAsync(
        d_keys.data(), keys.data(), nvalues * sizeof(std::int32_t), hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_values.data(),
                   values.data(),
                   nvalues * sizeof(std::int32_t),
                   hipMemcpyHostToDevice,
                   stream);
    state.ResumeTiming();

    xstd::hip::association_map<std::int32_t> associator(nvalues, 2u, stream);
    associator.fill(d_keys, d_values, stream);

    state.PauseTiming();
    hipStreamDestroy(stream);
    state.ResumeTiming();
  }
}

BENCHMARK(BM_BuildBinaryAssociatorHIP)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_BuildBinaryAssociatorHIPStreamed)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_MAIN();
