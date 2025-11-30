// SPDX-License-Identifier: MPL-2.0

/// @author Simone Balducci

#pragma once

#include <execution>

namespace xstd::internal {

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  inline constexpr auto default_policy = std::execution::par_unseq;
#else
  inline constexpr auto default_policy = std::execution::unseq;
#endif

}  // namespace xstd::internal
