
#pragma once

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

  public:
    explicit host_unique(std::remove_extent_t<T>* data, host::Deleter deleter)
        : m_data{data, deleter} {}

    auto* data() const { return m_data.get(); }

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

  template <typename T>
  auto make_host_unique() {
    static_assert(std::is_trivially_constructible<T>::value,
                  "Allocating with non-trivial constructor is not supported");
    T* buf;
    hipHostMalloc(&buf, sizeof(T));
    return typename detail::make_host_selector_t<T>{buf, host::Deleter{}};
  }

  template <typename T>
  auto make_host_unique(std::size_t size) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value,
                  "Allocating with non-trivial constructor is not supported");
    void* buf;
    hipHostMalloc(&buf, sizeof(element_type) * size);
    return typename detail::make_host_selector_t<T>{reinterpret_cast<element_type*>(buf),
                                                    host::Deleter{}};
  }

}  // namespace xstd::hip
