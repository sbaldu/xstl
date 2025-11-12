
#pragma once

#include <alpaka/alpaka.hpp>
#include <span>

namespace xstd {
  namespace alpaka {
    namespace detail {

      namespace alpaka = ::alpaka;

      template <typename TFunc>
      struct KernelComputeAssociations {
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      std::span<int32_t> associations,
                                      TFunc func) const {
          for (auto i : alpaka::uniformElements(acc, associations.size())) {
            associations[i] = func(i);
          }
        }
      };

      struct KernelComputeAssociationSizes {
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      std::span<const int32_t> associations,
                                      std::span<int32_t> bin_sizes) const {
          for (auto i : alpaka::uniformElements(acc, associations.size())) {
            if (associations[i] >= 0)
              alpaka::atomicAdd(acc, &bin_sizes[associations[i]], 1);
          }
        }
      };

      struct KernelFillAssociator {
        template <typename TAcc, typename T>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      std::span<T> indexes,
                                      std::span<const int32_t> keys_buffer,
                                      std::span<int32_t> temp_offsets) const {
          for (auto i : alpaka::uniformElements(acc, keys_buffer.size())) {
            const auto key = keys_buffer[i];
            if (key >= 0) {
              indexes[alpaka::atomicAdd(acc, &temp_offsets[key], 1)] = i;
            }
          }
        }
      };

    }  // namespace detail
  }  // namespace alpaka
}  // namespace xstd
