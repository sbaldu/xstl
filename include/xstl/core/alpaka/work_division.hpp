// SPDX-License-Identifier: MPL-2.0

/// @author Simone Balducci

#pragma once

#include "xstl/core/alpaka/defines.hpp"

namespace xstd {
  namespace alpaka {
    namespace internal {

      template <typename TAcc>
      inline WorkDiv<Dim1D> make_workdiv(Idx blocksPerGrid,
                                         Idx threadsPerBlockOrElementsPerThread) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        if constexpr (std::is_same_v<TAcc, alpaka::AccGpuCudaRt<Dim1D, Idx>>) {
          // On GPU backends, each thread is looking at a single element:
          //   - threadsPerBlockOrElementsPerThread is the number of threads per block;
          //   - elementsPerThread is always 1.
          const auto elementsPerThread = Idx{1};
          return WorkDiv<Dim1D>(
              blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
        } else
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#if ALPAKA_ACC_GPU_HIP_ENABLED
            if constexpr (std::is_same_v<TAcc, alpaka::AccGpuHipRt<Dim1D, Idx>>) {
          // On GPU backends, each thread is looking at a single element:
          //   - threadsPerBlockOrElementsPerThread is the number of threads per block;
          //   - elementsPerThread is always 1.
          const auto elementsPerThread = Idx{1};
          return WorkDiv<Dim1D>(
              blocksPerGrid, threadsPerBlockOrElementsPerThread, elementsPerThread);
        } else
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED
        {
          // On CPU backends, run serially with a single thread per block:
          //   - threadsPerBlock is always 1;
          //   - threadsPerBlockOrElementsPerThread is the number of elements per thread.
          const auto threadsPerBlock = Idx{1};
          return WorkDiv<Dim1D>(blocksPerGrid, threadsPerBlock, threadsPerBlockOrElementsPerThread);
        }
      }

    }  // namespace internal
  }  // namespace alpaka
}  // namespace xstd
