
#pragma once

#include "xstl/cpu/association_map.hpp"

#ifdef __CUDACC__
#include "xstl/cuda/association_map.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_GPU_CUDA_ENABLED) ||       \
    defined(ALPAKA_ACC_GPU_HIP_ENABLED) || defined(ALPAKA_ACC_SYCL_ENABLED)
#include "xstl/alpaka/association_map.hpp"
#include "xstl/alpaka/detail/defines.hpp"
#endif
