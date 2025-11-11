
#pragma once

#include <cuda_runtime.h>

namespace xstd::cuda {

  inline auto current_device() {
    int device;
    cudaGetDevice(&device);
    return device;
  }

}  // namespace xstd::cuda
