// SPDX-License-Identifier: MPL-2.0

/// @author Simone Balducci

#pragma once

namespace xstd {
  namespace alpaka {
    namespace detail {

      namespace alpaka = ::alpaka;

      template <typename TDev>
      struct keys_host_wrapper {
        using type = alpaka::Buf<alpaka::DevCpu, int32_t, internal::Dim1D, internal::Idx>;
      };

      template <>
      struct keys_host_wrapper<alpaka::DevCpu> {
        using type = alpaka::Buf<alpaka::DevCpu, int32_t, internal::Dim1D, internal::Idx>&;
      };

    }  // namespace detail
  }  // namespace alpaka
}  // namespace xstd
