// SPDX-License-Identifier: MPL-2.0

/// @file current_device.hpp
/// @brief A header file defining a function to get the current CUDA device
/// @author Simone Balducci

#pragma once

#include <cuda_runtime.h>

namespace xstd::cuda {

  inline auto current_device() {
    int device;
    cudaGetDevice(&device);
    return device;
  }

}  // namespace xstd::cuda
