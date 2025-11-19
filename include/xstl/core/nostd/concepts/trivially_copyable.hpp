/// @file trivially_constructible.hpp
/// @brief A header file defining the trivially_copyable concept
/// @author Simone Balducci

#pragma once

#include <type_traits>

namespace xstd::nostd {

  template <typename T>
  concept trivially_copyable = std::is_trivially_copyable_v<T>;

}
