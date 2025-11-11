
#pragma once

#include "xstl/core/cuda/device_unique.hpp"
#include "xstl/cuda/association_map.hpp"
#include "xstl/cuda/detail/kernels.cuh"
#include <span>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace xstd::cuda {

  template <typename T>
  inline void association_map<T>::fill_impl(cudaStream_t stream,
                                            std::span<key_type> keys,
                                            std::span<mapped_type> values) {
    auto accumulator = make_device_unique<key_type[]>(m_keys, stream);
    cudaMemsetAsync(accumulator.data(), 0, sizeof(key_type) * m_keys, stream);
    const auto block_size = 256u;
    const auto grid_size = (keys.size() + block_size - 1) / block_size;
    detail::KernelComputeAssociationSizes<<<grid_size, block_size, 0, stream>>>(
        keys.data(), accumulator.data(), values.size());

    auto temporary_keys = make_device_unique<key_type[]>(m_keys + 1, stream);
    cudaMemsetAsync(temporary_keys.data(), 0, sizeof(key_type) * (m_keys + 1), stream);
    cudaStreamSynchronize(stream);  // TODO: remove this sync by using async scan
    thrust::inclusive_scan(
        thrust::device, accumulator.data(), accumulator.data() + m_keys, temporary_keys.data() + 1);
    cudaMemcpyAsync(m_data.keys.data(),
                    temporary_keys.data(),
                    sizeof(key_type) * (m_keys + 1),
                    cudaMemcpyDeviceToDevice,
                    stream);
    detail::KernelFillAssociator<<<grid_size, block_size, 0, stream>>>(
        m_data.values.data(), keys.data(), temporary_keys.data(), values.size());
    cudaMemcpyAsync(m_data.keys_host.data(),
                    m_data.keys.data(),
                    sizeof(key_type) * (m_keys + 1),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
  }

}  // namespace xstd::cuda
