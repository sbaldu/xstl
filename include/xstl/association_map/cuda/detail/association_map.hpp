
#pragma once

#include "xstl/core/cuda/device_unique.hpp"
#include "xstl/association_map/cuda/association_map.hpp"
#include "xstl/association_map/cuda/detail/kernels.cuh"
#include <span>
#include <thrust/execution_policy.h>
#include <thrust/async/scan.h>
#include <thrust/scan.h>

namespace xstd::cuda {

  template <typename T>
  inline void association_map<T>::fill_impl(std::span<key_type> keys,
                                            std::span<mapped_type> values) {
    auto accumulator = make_device_unique<key_type[]>(m_extents.keys);
    cudaMemset(accumulator.data(), 0, sizeof(key_type) * m_extents.keys);
    const auto block_size = 256u;
    const auto grid_size = (keys.size() + block_size - 1) / block_size;
    detail::KernelComputeAssociationSizes<<<grid_size, block_size>>>(
        keys.data(), accumulator.data(), values.size());

    auto temporary_keys = make_device_unique<key_type[]>(m_extents.keys + 1);
    cudaMemset(temporary_keys.data(), 0, sizeof(key_type) * (m_extents.keys + 1));
    thrust::inclusive_scan(thrust::device,
                           accumulator.data(),
                           accumulator.data() + m_extents.keys,
                           temporary_keys.data() + 1);
    cudaMemcpy(m_data.keys.data(),
               temporary_keys.data(),
               sizeof(key_type) * (m_extents.keys + 1),
               cudaMemcpyDeviceToDevice);
    detail::KernelFillAssociator<<<grid_size, block_size>>>(
        m_data.values.data(), keys.data(), temporary_keys.data(), values.size());
    cudaMemcpy(m_data.keys_host.data(),
               m_data.keys.data(),
               sizeof(key_type) * (m_extents.keys + 1),
               cudaMemcpyDeviceToHost);
  }

  template <typename T>
  inline void association_map<T>::fill_impl(cudaStream_t stream,
                                            std::span<key_type> keys,
                                            std::span<mapped_type> values) {
    auto accumulator = make_device_unique<key_type[]>(m_extents.keys, stream);
    cudaMemsetAsync(accumulator.data(), 0, sizeof(key_type) * m_extents.keys, stream);
    const auto block_size = 256u;
    const auto grid_size = (keys.size() + block_size - 1) / block_size;
    detail::KernelComputeAssociationSizes<<<grid_size, block_size, 0, stream>>>(
        keys.data(), accumulator.data(), values.size());

    auto temporary_keys = make_device_unique<key_type[]>(m_extents.keys + 1, stream);
    cudaMemsetAsync(temporary_keys.data(), 0, sizeof(key_type) * (m_extents.keys + 1), stream);
    thrust::async::inclusive_scan(thrust::device.on(stream),
                                  accumulator.data(),
                                  accumulator.data() + m_extents.keys,
                                  temporary_keys.data() + 1);
    cudaMemcpyAsync(m_data.keys.data(),
                    temporary_keys.data(),
                    sizeof(key_type) * (m_extents.keys + 1),
                    cudaMemcpyDeviceToDevice,
                    stream);
    detail::KernelFillAssociator<<<grid_size, block_size, 0, stream>>>(
        m_data.values.data(), keys.data(), temporary_keys.data(), values.size());
    cudaMemcpyAsync(m_data.keys_host.data(),
                    m_data.keys.data(),
                    sizeof(key_type) * (m_extents.keys + 1),
                    cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
  }

}  // namespace xstd::cuda
