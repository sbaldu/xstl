/// @file device_unique.hpp
/// @brief A header file defining a unique pointer for HIP device memory
/// @author Simone Balducci, CMS-Patatrack team (CERN)

#pragma once

#include "xstl/core/hip/current_device.hpp"
#include "xstl/core/nostd/concepts/trivially_constructible.hpp"
#include <hip_runtime.h>

#include <memory>
#include <type_traits>

namespace xstd::hip {

  namespace device {

    class Deleter {
    private:
      int m_device;

    public:
      Deleter() = default;
      Deleter(int device) : m_device{device} {}

      void operator()(void* ptr) { hipFree(ptr); }
    };

  }  // namespace device

  template <typename T>
  class device_unique {
  private:
    std::unique_ptr<T, device::Deleter> m_data;
    std::size_t m_size;

  public:
    explicit device_unique(std::remove_extent_t<T>* data, std::size_t size, device::Deleter deleter)
        : m_data{data, deleter}, m_size{size} {}
    auto* data() const { return m_data.get(); }
    auto size() const { return m_size; }

    operator std::span<const T>() const { return std::span<const T>{m_data.get(), m_size}; }
    operator std::span<T>() { return std::span<T>{m_data.get(), m_size}; }
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

  template <nostd::trivially_constructible T>
  auto make_device_unique() {
    auto dev = current_device();
    T* buf;
    hipMalloc(&buf, sizeof(T));
    return typename detail::make_device_selector_t<T>{buf, 1, device::Deleter{dev}};
  }

  template <nostd::trivially_constructible T>
  auto make_device_unique(std::size_t size) {
    using element_type = typename std::remove_extent<T>::type;
    auto dev = current_device();
    void* buf;
    hipMalloc(&buf, sizeof(element_type) * size);
    return typename detail::make_device_selector_t<T>{reinterpret_cast<element_type*>(buf),
                                                      size device::Deleter{dev}};
  }

  template <nostd::trivially_constructible T>
  auto make_device_unique(hipStream_t stream) {
    auto dev = current_device();
    T* buf;
    hipMallocAsync(&buf, sizeof(T), stream);
    return typename detail::make_device_selector_t<T>{buf, 1, device::Deleter{dev}};
  }

  template <nostd::trivially_constructible T>
  auto make_device_unique(std::size_t size, hipStream_t stream) {
    using element_type = typename std::remove_extent<T>::type;
    auto dev = current_device();
    void* buf;
    hipMallocAsync(&buf, sizeof(element_type) * size, stream);
    return typename detail::make_device_selector_t<T>{
        reinterpret_cast<element_type*>(buf), size, device::Deleter{dev}};
  }

}  // namespace xstd::hip
