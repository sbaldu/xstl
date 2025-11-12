
#pragma once

#include <hip_runtime.h>
#include <span>

namespace xstd::hip::detail {

  __global__ void KernelComputeAssociationSizes(std::span<const int32_t> associations,
                                                std::span<int32_t> bin_sizes) {
    const auto thidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thidx < associations.size() and associations[thidx] >= 0) {
      atomicAdd(&bin_sizes[associations[thidx]], 1);
    }
  }

  template <typename T>
  __global__ void KernelFillAssociator(std::span<T> indexes,
                                       std::span<const int32_t> keys_buffer,
                                       std::span<int32_t> temp_offsets) {
    const auto thidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thidx < keys_buffer.size()) {
      const auto key = keys_buffer[thidx];
      if (key >= 0) {
        indexes[atomicAdd(&temp_offsets[bin_id], 1)] = thidx;
      }
    }
  }

}  // namespace xstd::hip::detail
