
#include "xstl/xstl.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <ranges>

static void BM_BuildBinaryAssociatorCUDA(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto nvalues = state.range(0);
    auto values = xstd::cuda::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::copy(std::views::iota(0) | std::views::take(nvalues), values.data());
    auto keys = xstd::cuda::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::transform(std::views::iota(0) | std::views::take(nvalues),
                           keys.data(),
                           [](auto x) -> int32_t { return x % 2 == 0; });
    auto d_keys = xstd::cuda::make_device_unique<std::int32_t[]>(nvalues);
    auto d_values = xstd::cuda::make_device_unique<std::int32_t[]>(nvalues);
    cudaMemcpy(d_keys.data(), keys.data(), nvalues * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(
        d_values.data(), values.data(), nvalues * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    state.ResumeTiming();

    xstd::cuda::association_map<std::int32_t> associator(nvalues, 2u);
    associator.fill(d_keys, d_values);
  }
}

static void BM_BuildBinaryAssociatorCUDAStreamed(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    const auto nvalues = state.range(0);
    auto values = xstd::cuda::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::copy(std::views::iota(0) | std::views::take(nvalues), values.data());
    auto keys = xstd::cuda::make_host_unique<std::int32_t[]>(nvalues);
    std::ranges::transform(std::views::iota(0) | std::views::take(nvalues),
                           keys.data(),
                           [](auto x) -> int32_t { return x % 2 == 0; });

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto d_keys = xstd::cuda::make_device_unique<std::int32_t[]>(nvalues, stream);
    auto d_values = xstd::cuda::make_device_unique<std::int32_t[]>(nvalues, stream);
    cudaMemcpyAsync(
        d_keys.data(), keys.data(), nvalues * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values.data(),
                    values.data(),
                    nvalues * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice,
                    stream);
    state.ResumeTiming();

    xstd::cuda::association_map<std::int32_t> associator(nvalues, 2u, stream);
    associator.fill(d_keys, d_values, stream);

    state.PauseTiming();
    cudaStreamDestroy(stream);
    state.ResumeTiming();
  }
}

BENCHMARK(BM_BuildBinaryAssociatorCUDA)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_BuildBinaryAssociatorCUDAStreamed)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);
BENCHMARK_MAIN();
