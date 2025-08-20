
#pragma once

#include <cstdint>
#include <span>

namespace xstd {
  namespace internal {

    // map_interface provides a common interface for map-like structures, enforcing the
    // implementations of all the common methods accross every backend.
    template <typename TMap>
    struct map_interface {
      auto empty() const { return static_cast<const TMap*>(this)->m_nvalues == 0; }
      auto size() const { return static_cast<const TMap*>(this)->m_nvalues; }
      auto extents() const { return static_cast<const TMap*>(this)->extents_impl(); }

      auto begin() { return static_cast<TMap&>(*this).m_data.values.data(); }
      auto begin() const { return static_cast<TMap&>(*this).m_data.values.data(); }
      auto cbegin() const { return static_cast<TMap&>(*this).m_data.values.data(); }

      auto end() {
        auto& m = static_cast<TMap&>(*this);
        return m.m_data.values.data() + m.m_nvalues;
      }
      auto end() const {
        auto& m = static_cast<TMap&>(*this);
        return m.m_data.values.data() + m.m_nvalues;
      }
      auto cend() const {
        auto& m = static_cast<TMap&>(*this);
        return m.m_data.values.data() + m.m_nvalues;
      }

      template <typename... TArgs>
      auto find(TArgs&&... args) {
        return static_cast<TMap*>(this)->find_impl(std::forward<TArgs>(args)...);
      }
      template <typename... TArgs>
      auto find(TArgs&&... args) const {
        return static_cast<const TMap*>(this)->find_impl(std::forward<TArgs>(args)...);
      }

      template <typename... TArgs>
      auto count(TArgs&&... args) const {
        static_assert(std::is_invocable_v<decltype(&TMap::count_impl), const TMap, TArgs...>,
                      "The arguments provided are not compatible with the selected backend.");
        return static_cast<const TMap*>(this)->count_impl(std::forward<TArgs>(args)...);
      }

      template <typename... TArgs>
      bool contains(TArgs&&... args) const {
        static_assert(std::is_invocable_v<decltype(&TMap::contains_impl), const TMap, TArgs...>,
                      "The arguments provided are not compatible with the selected backend.");
        return static_cast<const TMap*>(this)->contains_impl(std::forward<TArgs>(args)...);
      }

      template <typename... TArgs>
      auto lower_bound(TArgs&&... args) {
        return static_cast<TMap*>(this)->lower_bound_impl(std::forward<TArgs>(args)...);
      }
      template <typename... TArgs>
      auto lower_bound(TArgs&&... args) const {
        return static_cast<const TMap*>(this)->lower_bound_impl(std::forward<TArgs>(args)...);
      }

      template <typename... TArgs>
      auto upper_bound(TArgs&&... args) {
        return static_cast<TMap*>(this)->upper_bound_impl(std::forward<TArgs>(args)...);
      }
      template <typename... TArgs>
      auto upper_bound(TArgs&&... args) const {
        return static_cast<const TMap*>(this)->upper_bound_impl(std::forward<TArgs>(args)...);
      }

      template <typename... TArgs>
      auto equal_range(TArgs&&... args) {
        return static_cast<TMap*>(this)->equal_range_impl(std::forward<TArgs>(args)...);
      }
      template <typename... TArgs>
      auto equal_range(TArgs&&... args) const {
        return static_cast<const TMap*>(this)->equal_range_impl(std::forward<TArgs>(args)...);
      }

      template <typename... TArgs>
      void fill(TArgs&&... args) {
        static_assert(std::is_invocable_v<decltype(&TMap::fill_impl), TMap, TArgs...>,
                      "The arguments provided are not compatible with the selected backend.");
        static_cast<TMap*>(this)->fill_impl(std::forward<TArgs>(args)...);
      }

      auto* view() { return static_cast<TMap*>(this)->view_impl(); }
    };

  }  // namespace internal
}  // namespace xstd
