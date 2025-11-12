/// @file trivially_constructible.hpp
/// @brief A header file defining the trivially_constructible concept
/// @author Simone Balducci

#pragma once

#include <type_traits>

namespace xstd::nostd {

  template <typename T>
  concept trivially_copiable = std::is_trivially_constructible_v<T>;

}
