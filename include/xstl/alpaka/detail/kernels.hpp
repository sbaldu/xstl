
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
                                      int32_t* associations,
                                      TFunc func,
                                      std::size_t values_size) const {
          for (auto i : alpaka::uniformElements(acc, values_size)) {
            associations[i] = func(i);
          }
        }
      };

      struct KernelComputeAssociationSizes {
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      const int32_t* associations,
                                      int32_t* bin_sizes,
                                      std::size_t values_size) const {
          for (auto i : alpaka::uniformElements(acc, values_size)) {
            if (associations[i] >= 0)
              alpaka::atomicAdd(acc, &bin_sizes[associations[i]], 1);
          }
        }
      };

      struct KernelFillAssociator {
        template <typename TAcc, typename TMapped>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      TMapped* indexes,
                                      const int32_t* keys_buffer,
                                      int32_t* temp_offsets,
                                      std::size_t values_size) const {
          for (auto i : alpaka::uniformElements(acc, values_size)) {
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
