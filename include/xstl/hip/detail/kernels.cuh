
#pragma once

#include <hip_runtime.h>

namespace xstd::hip::detail {

  __global__ void KernelComputeAssociationSizes(const int32_t* associations,
                                                int32_t* bin_sizes,
                                                std::size_t size) {
    const auto thidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thidx < size and associations[thidx] >= 0) {
      atomicAdd(&bin_sizes[associations[thidx]], 1);
    }
  }

  __global__ void KernelFillAssociator(int32_t* indexes,
                                       const int32_t* bin_buffer,
                                       int32_t* temp_offsets,
                                       std::size_t size) {
    const auto thidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thidx < size) {
      const auto bin_id = bin_buffer[thidx];
      if (bin_id >= 0) {
        auto prev = atomicAdd(&temp_offsets[bin_id], 1);
        indexes[prev] = thidx;
      }
    }
  }

}  // namespace xstd::hip::detail
