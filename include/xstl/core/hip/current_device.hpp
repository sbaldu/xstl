/// @file current_device.hpp
/// @brief A header file defining a function to get the current HIP device
/// @author Simone Balducci

#pragma once

#include <hip_runtime.h>

namespace xstd::hip {

  inline auto current_device() {
    int device;
    hipGetDevice(&device);
    return device;
  }

}  // namespace xstd::hip
