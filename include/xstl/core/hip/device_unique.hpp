
#pragma once

#include "xstl/core/hip/current_device.hpp"
#include <hip_runtime.h>

#include <memory>
#include <type_traits>

namespace xstd::hip {

  class Deleter {
  private:
    int m_device;

  public:
    Deleter() = default;
    Deleter(int device) : m_device{device} {}

    void operator()(void* ptr) { hipFree(ptr); }
  };

  template <typename T>
  class device_unique {
  private:
    std::unique_ptr<T, Deleter> m_data;

  public:
    explicit device_unique(std::remove_extent_t<T>* data, Deleter deleter)
        : m_data{data, deleter} {}
    auto* data() const { return m_data.get(); }
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

  template <typename T>
  auto make_device_unique() {
    static_assert(std::is_trivially_constructible<T>::value,
                  "Allocating with non-trivial constructor on the device memory is not supported");
    auto dev = current_device();
    T* buf;
    hipMalloc(&buf, sizeof(T));
    return typename detail::make_device_selector_t<T>{buf, Deleter{dev}};
  }

  template <typename T>
  auto make_device_unique(std::size_t size) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value,
                  "Allocating with non-trivial constructor on the device memory is not supported");
    auto dev = current_device();
    void* buf;
    hipMalloc(&buf, sizeof(element_type) * size);
    return typename detail::make_device_selector_t<T>{reinterpret_cast<element_type*>(buf),
                                                      Deleter{dev}};
  }

  template <typename T>
  auto make_device_unique(hipStream_t stream) {
    static_assert(std::is_trivially_constructible<T>::value,
                  "Allocating with non-trivial constructor on the device memory is not supported");
    auto dev = current_device();
    T* buf;
    hipMallocAsync(&buf, sizeof(T), stream);
    return typename detail::make_device_selector_t<T>{buf, Deleter{dev}};
  }

  template <typename T>
  auto make_device_unique(std::size_t size, hipStream_t stream) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value,
                  "Allocating with non-trivial constructor on the device memory is not supported");
    auto dev = current_device();
    void* buf;
    hipMallocAsync(&buf, sizeof(element_type) * size, stream);
    return typename detail::make_device_selector_t<T>{reinterpret_cast<element_type*>(buf),
                                                      Deleter{dev}};
  }

}  // namespace xstd::hip
