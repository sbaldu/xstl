// SPDX-License-Identifier: MPL-2.0

/// @author Simone Balducci

#pragma once

#include <cuda_runtime.h>
#include <span>

namespace xstd::cuda::detail {

  __global__ void KernelComputeAssociationSizes(const int32_t* associations,
                                                int32_t* bin_sizes,
                                                std::size_t values_size) {
    const auto thidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thidx < values_size and associations[thidx] >= 0) {
      atomicAdd(&bin_sizes[associations[thidx]], 1);
    }
  }

  template <typename T>
  __global__ void KernelFillAssociator(T* indexes,
                                       const int32_t* keys_buffer,
                                       int32_t* temp_offsets,
                                       std::size_t values_size) {
    const auto thidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thidx < values_size) {
      const auto key = keys_buffer[thidx];
      if (key >= 0) {
        indexes[atomicAdd(&temp_offsets[key], 1)] = thidx;
      }
    }
  }

}  // namespace xstd::cuda::detail
