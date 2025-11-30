// SPDX-License-Identifier: MPL-2.0

/// @file device_unique.hpp
/// @brief A header file defining a unique pointer for HIP host pinned memory
/// @author Simone Balducci, CMS-Patatrack team (CERN)

#pragma once

#include "xstl/core/nostd/concepts/trivially_copyable.hpp"
#include <hip_runtime.h>
#include <memory>

namespace xstd::hip {

  namespace host {

    struct Deleter {
      Deleter() = default;

      void operator()(void* ptr) { hipHostFree(ptr); }
    };

  }  // namespace host

  template <typename T>
  class host_unique {
  private:
    std::unique_ptr<T, host::Deleter> m_data;
    std::size_t m_size;

  public:
    using value_type = std::remove_extent_t<T>;

    explicit host_unique(value_type* data, std::size_t size, host::Deleter deleter)
        : m_data{data, deleter}, m_size{size} {}

    auto* data() const { return m_data.get(); }
    auto size() const { return m_size; }

    const auto& operator[](std::size_t idx) const { return m_data[idx]; }
    auto& operator[](std::size_t idx) { return m_data[idx]; }

    operator std::span<const value_type>() const {
      return std::span<const value_type>{m_data.get(), m_size};
    }
    operator std::span<value_type>() { return std::span<value_type>{m_data.get(), m_size}; }
  };

  namespace detail {

    template <typename T>
    struct make_host_selector {
      using type = host_unique<T>;
    };
    template <typename T>
    struct make_host_selector<T[]> {
      using type = host_unique<T[]>;
    };

    template <typename T>
    using make_host_selector_t = make_host_selector<T>::type;

  }  // namespace detail

  template <nostd::trivially_copyable T>
  auto make_host_unique() {
    T* buf;
    hipHostMalloc(&buf, sizeof(T));
    return typename detail::make_host_selector_t<T>{buf, 1, host::Deleter{}};
  }

  template <nostd::trivially_copyable T>
  auto make_host_unique(std::size_t size) {
    using element_type = typename std::remove_extent<T>::type;
    void* buf;
    hipHostMalloc(&buf, sizeof(element_type) * size);
    return typename detail::make_host_selector_t<T>{
        reinterpret_cast<element_type*>(buf), size, host::Deleter{}};
  }

}  // namespace xstd::hip
