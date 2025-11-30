// SPDX-License-Identifier: MPL-2.0

/// @file association_map/association_map.hpp
/// @brief Geader file including all the implementations of the association_map
/// @author Simone Balducci

#pragma once

#include "xstl/association_map/cpu/association_map.hpp"

#ifdef __CUDACC__
#include "xstl/association_map/cuda/association_map.hpp"
#endif

#ifdef __HIPCC__
#include "xstl/association_map/hip/association_map.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED) ||       \
    defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_SYCL_ENABLED)
#include "xstl/association_map/alpaka/association_map.hpp"
#include "xstl/core/alpaka/defines.hpp"
#endif
