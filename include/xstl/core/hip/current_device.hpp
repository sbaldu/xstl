
#pragma once

#include <hip_runtime.h>

namespace xstd::hip {

  inline auto current_device() {
    int device;
    hipGetDevice(&device);
    return device;
  }

}  // namespace xstd::hip
