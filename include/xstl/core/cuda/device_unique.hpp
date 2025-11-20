/// @file device_unique.hpp
/// @brief A header file defining a unique pointer for CUDA device memory
/// @author Simone Balducci, CMS-Patatrack team (CERN)

#pragma once

#include "xstl/core/cuda/current_device.hpp"
#include "xstl/core/nostd/concepts/trivially_copyable.hpp"
#include <cuda_runtime.h>

#include <memory>
#include <type_traits>

namespace xstd::cuda {

  namespace device {

    class Deleter {
    private:
      int m_device;

    public:
      Deleter() = default;
      Deleter(int device) : m_device{device} {}

      void operator()(void* ptr) { cudaFree(ptr); }
    };

  }  // namespace device

  template <typename T>
  class device_unique {
  private:
    std::unique_ptr<T, device::Deleter> m_data;
    std::size_t m_size;

  public:
    using value_type = std::remove_extent_t<T>;

    explicit device_unique(value_type* data, std::size_t size, device::Deleter deleter)
        : m_data{data, deleter}, m_size{size} {}
    auto* data() const { return m_data.get(); }
    auto size() const { return m_size; }

    operator std::span<const value_type>() const {
      return std::span<const value_type>{m_data.get(), m_size};
    }
    operator std::span<value_type>() { return std::span<value_type>{m_data.get(), m_size}; }
  };

  namespace detail {

    template <typename T>
    struct make_device_selector {
      using type = device_unique<T>;
    };
    template <typename T>
    struct make_device_selector<T[]> {
      using type = device_unique<T[]>;
    };

    template <typename T>
    using make_device_selector_t = make_device_selector<T>::type;

  }  // namespace detail

  template <nostd::trivially_copyable T>
  auto make_device_unique() {
    auto dev = current_device();
    T* buf;
    cudaMalloc(&buf, sizeof(T));
    return typename detail::make_device_selector_t<T>{buf, 1, device::Deleter{dev}};
  }

  template <nostd::trivially_copyable T>
  auto make_device_unique(std::size_t size) {
    using element_type = typename std::remove_extent<T>::type;
    auto dev = current_device();
    void* buf;
    cudaMalloc(&buf, sizeof(element_type) * size);
    return typename detail::make_device_selector_t<T>{
        reinterpret_cast<element_type*>(buf), size, device::Deleter{dev}};
  }

  template <nostd::trivially_copyable T>
  auto make_device_unique(cudaStream_t stream) {
    auto dev = current_device();
    T* buf;
    cudaMallocAsync(&buf, sizeof(T), stream);
    return typename detail::make_device_selector_t<T>{buf, 1, device::Deleter{dev}};
  }

  template <nostd::trivially_copyable T>
  auto make_device_unique(std::size_t size, cudaStream_t stream) {
    using element_type = typename std::remove_extent<T>::type;
    auto dev = current_device();
    void* buf;
    cudaMallocAsync(&buf, sizeof(element_type) * size, stream);
    return typename detail::make_device_selector_t<T>{
        reinterpret_cast<element_type*>(buf), size, device::Deleter{dev}};
  }

}  // namespace xstd::cuda
