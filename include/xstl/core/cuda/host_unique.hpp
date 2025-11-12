/// @file device_unique.hpp
/// @brief A header file defining a unique pointer for CUDA host pinned memory
/// @author Simone Balducci, CMS-Patatrack team (CERN)

#pragma once

#include "xstl/core/nostd/concepts/trivially_constructible.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace xstd::cuda {

  namespace host {

    struct Deleter {
      Deleter() = default;

      void operator()(void* ptr) { cudaFreeHost(ptr); }
    };

  }  // namespace host

  template <typename T>
  class host_unique {
  private:
    std::unique_ptr<T, host::Deleter> m_data;
    std::size_t m_size;

  public:
    explicit host_unique(std::remove_extent_t<T>* data, host::Deleter deleter)
        : m_data{data, deleter}, m_size{size} {}

    auto* data() const { return m_data.get(); }
    auto size() const { return m_size; }

    const auto& operator[](std::size_t idx) const { return m_data[idx]; }
    auto& operator[](std::size_t idx) { return m_data[idx]; }
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

  template <nostd::trivially_constructible T>
  auto make_host_unique() {
    T* buf;
    cudaMallocHost(&buf, sizeof(T));
    return typename detail::make_host_selector_t<T>{buf, 1, host::Deleter{}};
  }

  template <nostd::trivially_constructible T>
  auto make_host_unique(std::size_t size) {
    using element_type = typename std::remove_extent<T>::type;
    void* buf;
    cudaMallocHost(&buf, sizeof(element_type) * size);
    return typename detail::make_host_selector_t<T>{
        reinterpret_cast<element_type*>(buf), size, host::Deleter{}};
  }

}  // namespace xstd::cuda
