
#pragma once

#include "xstl/core/hip/device_unique.hpp"
#include "xstl/hip/association_map.hpp"
#include "xstl/hip/detail/kernels.cuh"
#include <span>
#include <thrust/execution_policy.h>
#include <thrust/async/scan.h>
#include <thrust/scan.h>

namespace xstd::hip {

  template <typename T>
  inline void association_map<T>::fill_impl(std::span<key_type> keys,
                                            std::span<mapped_type> values) {
    auto accumulator = make_device_unique<key_type[]>(m_extents.keys);
    hipMemset(accumulator.data(), 0, sizeof(key_type) * m_extents.keys);
    const auto block_size = 256u;
    const auto grid_size = (keys.size() + block_size - 1) / block_size;
    detail::KernelComputeAssociationSizes<<<grid_size, block_size>>>(keys, accumulator);

    auto temporary_keys = make_device_unique<key_type[]>(m_extents.keys + 1);
    hipMemset(temporary_keys.data(), 0, sizeof(key_type) * (m_extents.keys + 1));
    thrust::inclusive_scan(thrust::device,
                           accumulator.data(),
                           accumulator.data() + m_extents.keys,
                           temporary_keys.data() + 1);
    hipMemcpy(m_data.keys.data(),
              temporary_keys.data(),
              sizeof(key_type) * (m_extents.keys + 1),
              hipMemcpyDeviceToDevice);
    detail::KernelFillAssociator<<<grid_size, block_size>>>(m_data.values, keys, temporary_keys);
    hipMemcpy(m_data.keys_host.data(),
              m_data.keys.data(),
              sizeof(key_type) * (m_extents.keys + 1),
              hipMemcpyDeviceToHost);
  }

  template <typename T>
  inline void association_map<T>::fill_impl(hipStream_t stream,
                                            std::span<key_type> keys,
                                            std::span<mapped_type> values) {
    auto accumulator = make_device_unique<key_type[]>(m_extents.keys, stream);
    hipMemsetAsync(accumulator.data(), 0, sizeof(key_type) * m_extents.keys, stream);
    const auto block_size = 256u;
    const auto grid_size = (keys.size() + block_size - 1) / block_size;
    detail::KernelComputeAssociationSizes<<<grid_size, block_size, 0, stream>>>(keys, accumulator);

    auto temporary_keys = make_device_unique<key_type[]>(m_extents.keys + 1, stream);
    hipMemsetAsync(temporary_keys.data(), 0, sizeof(key_type) * (m_extents.keys + 1), stream);
    thrust::async::inclusive_scan(thrust::device.on(stream),
                                  accumulator.data(),
                                  accumulator.data() + m_extents.keys,
                                  temporary_keys.data() + 1);
    hipMemcpyAsync(m_data.keys.data(),
                   temporary_keys.data(),
                   sizeof(key_type) * (m_extents.keys + 1),
                   hipMemcpyDeviceToDevice,
                   stream);
    detail::KernelFillAssociator<<<grid_size, block_size, 0, stream>>>(
        m_data.values, keys, temporary_keys);
    hipMemcpyAsync(m_data.keys_host.data(),
                   m_data.keys.data(),
                   sizeof(key_type) * (m_extents.keys + 1),
                   hipMemcpyDeviceToHost,
                   stream);
    hipStreamSynchronize(stream);
  }

}  // namespace xstd::hip
