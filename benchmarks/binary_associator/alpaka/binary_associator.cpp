
#include "xstl/xstl.hpp"
#include <benchmark/benchmark.h>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <ranges>

static void BM_BuildBinaryAssociatorAlpaka(benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();
    auto host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
    auto device = alpaka::getDevByIdx(xstd::alpaka::Platform{}, 0u);
    auto queue = xstd::alpaka::Queue(device);

    const auto nvalues = static_cast<std::size_t>(state.range(0));
    auto h_values =
        alpaka::allocMappedBuf<std::int32_t, std::size_t>(host, alpaka::PlatformCpu{}, nvalues);
    std::ranges::copy(std::views::iota(0) | std::views::take(nvalues), h_values.data());
    auto h_keys =
        alpaka::allocMappedBuf<std::int32_t, std::size_t>(host, alpaka::PlatformCpu{}, nvalues);
    std::ranges::transform(std::views::iota(0) | std::views::take(nvalues),
                           h_keys.data(),
                           [](auto x) -> std::int32_t { return x % 2 == 0; });
    auto d_keys = alpaka::allocAsyncBuf<std::int32_t, std::size_t>(queue, nvalues);
    auto d_values = alpaka::allocAsyncBuf<std::int32_t, std::size_t>(queue, nvalues);
    alpaka::memcpy(queue, d_keys, h_keys);
    alpaka::memcpy(queue, d_values, h_values);
    state.ResumeTiming();

    xstd::alpaka::association_map<std::int32_t> associator(nvalues, 2u, queue);
    associator.fill(queue, std::span{d_keys.data(), nvalues}, std::span{d_values.data(), nvalues});
    alpaka::wait(queue);
  }
}

BENCHMARK(BM_BuildBinaryAssociatorAlpaka)->RangeMultiplier(2)->Range(1 << 10, 1 << 20);

BENCHMARK_MAIN();
