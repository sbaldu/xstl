
#pragma once

#include <cstdint>
#include <span>

namespace xstd {
  namespace internal {

    // map_interface provides a common interface for map-like structures, enforcing the
    // implementations of all the common methods accross every backend.
    template <typename TMap>
    struct map_interface {
      auto empty() const { return static_cast<const TMap*>(this)->m_values == 0; }
      auto size() const { return static_cast<const TMap*>(this)->m_values; }
      auto extents() const { return static_cast<const TMap*>(this)->extents_impl(); }

      auto begin() { return static_cast<TMap&>(*this).m_data.values.data(); }
      auto begin() const { return static_cast<TMap&>(*this).m_data.values.data(); }
      auto cbegin() const { return static_cast<TMap&>(*this).m_data.values.data(); }

      auto end() {
        auto& m = static_cast<TMap&>(*this);
        return m.m_data.values.data() + m.m_values;
      }
      auto end() const {
        auto& m = static_cast<TMap&>(*this);
        return m.m_data.values.data() + m.m_values;
      }
      auto cend() const {
        auto& m = static_cast<TMap&>(*this);
        return m.m_data.values.data() + m.m_values;
      }

      auto find(auto key) { return static_cast<TMap*>(this)->find_impl(key); }
      auto find(auto key) const { return static_cast<const TMap*>(this)->find_impl(key); }

      auto count(auto key) const { return static_cast<const TMap*>(this)->count_impl(key); }

      bool contains(auto key) const { return static_cast<const TMap*>(this)->contains_impl(key); }

      auto lower_bound(auto key) { return static_cast<TMap*>(this)->lower_bound_impl(key); }
      auto lower_bound(auto key) const {
        return static_cast<const TMap*>(this)->lower_bound_impl(key);
      }

      auto upper_bound(auto key) { return static_cast<TMap*>(this)->upper_bound_impl(key); }
      auto upper_bound(auto key) const {
        return static_cast<const TMap*>(this)->upper_bound_impl(key);
      }

      auto equal_range(auto key) { return static_cast<TMap*>(this)->equal_range_impl(key); }
      auto equal_range(auto key) const {
        return static_cast<const TMap*>(this)->equal_range_impl(key);
      }

      template <typename... TArgs>
      void fill(TArgs&&... args) {
        static_assert(
            requires(TMap& m, TArgs&&... xs) { m.fill_impl(std::forward<TArgs>(xs)...); },
            "The arguments provided are not compatible with the selected backend.");
        static_cast<TMap*>(this)->fill_impl(std::forward<TArgs>(args)...);
      }

      auto view() { return static_cast<TMap*>(this)->m_view; }
    };

  }  // namespace internal
}  // namespace xstd
