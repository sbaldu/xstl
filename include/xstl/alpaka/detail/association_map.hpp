

#pragma once

#include "xstl/internal/map_interface.hpp"
#include "xstl/alpaka/detail/defines.hpp"
#include "xstl/alpaka/detail/kernels.hpp"
#include "xstl/alpaka/internal/scan.hpp"
#include "xstl/alpaka/internal/work_division.hpp"
#include <alpaka/alpaka.hpp>

namespace xstd {
  namespace alpaka {

    template <typename T>
    association_map<T>::Extents association_map<T>::extents_impl() const {
      return {m_values, m_keys};
    }

    template <typename T>
    template <typename TQueue>
    inline void association_map<T>::fill_impl(TQueue& queue,
                                              std::span<association_map<T>::key_type> keys,
                                              std::span<association_map<T>::mapped_type> values) {
      auto accumulator =
          ::alpaka::allocAsyncBuf<key_type, internal::Idx>(queue, internal::Vec1D{m_keys});
      ::alpaka::memset(queue, accumulator, 0);
      const auto block_size = 256u;
      const auto grid_size = (keys.size() + block_size - 1) / block_size;
      const auto work_division = internal::make_workdiv<internal::Acc>(grid_size, block_size);
      ::alpaka::exec<internal::Acc>(queue,
                                    work_division,
                                    detail::KernelComputeAssociationSizes{},
                                    keys.data(),
                                    accumulator.data(),
                                    values.size());
      auto temporary_keys =
          ::alpaka::allocAsyncBuf<key_type, internal::Idx>(queue, internal::Vec1D{m_keys + 1});
      ::alpaka::memset(queue, temporary_keys, 0);
      internal::inclusive_scan(
          accumulator.data(), accumulator.data() + m_keys, temporary_keys.data() + 1);
      ::alpaka::memcpy(queue, m_data.keys, temporary_keys);
      ::alpaka::exec<internal::Acc>(queue,
                                    work_division,
                                    detail::KernelFillAssociator{},
                                    m_data.values.data(),
                                    keys.data(),
                                    temporary_keys.data(),
                                    values.size());
      ::alpaka::memcpy(queue, m_data.keys_host, m_data.keys);
      ::alpaka::wait(queue);
    }

  }  // namespace alpaka
}  // namespace xstd
