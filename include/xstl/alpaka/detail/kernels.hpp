
#pragma once

#include <alpaka/alpaka.hpp>

namespace xstd {
  namespace alpaka {
    namespace detail {

      template <typename TFunc>
      struct KernelComputeAssociations {
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      size_t size,
                                      int32_t* associations,
                                      TFunc func) const {
          for (auto i : ::alpaka::uniformElements(acc, size)) {
            associations[i] = func(i);
          }
        }
      };

      struct KernelComputeAssociationSizes {
        template <typename TAcc>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      const int32_t* associations,
                                      int32_t* bin_sizes,
                                      size_t size) const {
          for (auto i : ::alpaka::uniformElements(acc, size)) {
            if (associations[i] >= 0)
              ::alpaka::atomicAdd(acc, &bin_sizes[associations[i]], 1);
          }
        }
      };

      struct KernelFillAssociator {
        template <typename TAcc, typename T>
        ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                      T* indexes,
                                      const int32_t* bin_buffer,
                                      int32_t* temp_offsets,
                                      size_t size) const {
          for (auto i : ::alpaka::uniformElements(acc, size)) {
            const auto binId = bin_buffer[i];
            if (binId >= 0) {
              auto prev = ::alpaka::atomicAdd(acc, &temp_offsets[binId], 1);
              indexes[prev] = i;
            }
          }
        }
      };

    }  // namespace detail
  }  // namespace alpaka
}  // namespace xstd
