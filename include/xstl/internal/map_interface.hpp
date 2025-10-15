
#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>

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

      auto find(auto key) {
        auto& m = static_cast<TMap&>(*this);
        return typename TMap::iterator{m.m_data.values.data() + m.m_data.keys_host[key]};
      }
      auto find(auto key) const {
        const auto& m = static_cast<const TMap&>(*this);
        return typename TMap::const_iterator{m.m_data.values.data() + m.m_data.keys_host[key]};
      }

      auto count(auto key) const {
        const auto& m = static_cast<const TMap&>(*this);

        if (key < 0 || static_cast<typename TMap::size_type>(key) >= m.m_keys)
          throw std::out_of_range("Key is out of range.");
        return m.m_data.keys_host[key + 1] - m.m_data.keys_host[key];
      }

      bool contains(auto key) const {
        const auto& m = static_cast<const TMap&>(*this);

        if (key < 0 || static_cast<typename TMap::size_type>(key) >= m.m_keys)
          throw std::out_of_range("Key is out of range.");
        return m.m_data.keys_host[key + 1] > m.m_data.keys_host[key];
      }

      auto lower_bound(auto key) {
        auto& m = static_cast<TMap&>(*this);
        return typename TMap::iterator{m.m_data.values.data() + m.m_data.keys_host[key]};
      }
      auto lower_bound(auto key) const {
        const auto& m = static_cast<const TMap&>(*this);
        return typename TMap::const_iterator{m.m_data.values.data() + m.m_data.keys_host[key]};
      }

      auto upper_bound(auto key) {
        auto& m = static_cast<TMap&>(*this);
        return typename TMap::iterator{m.m_data.values.data() + m.m_data.keys_host[key + 1]};
      }
      auto upper_bound(auto key) const {
        const auto& m = static_cast<const TMap&>(*this);
        return typename TMap::const_iterator{m.m_data.values.data() + m.m_data.keys_host[key + 1]};
      }

      auto equal_range(auto key) {
        auto& m = static_cast<TMap&>(*this);
        return std::make_pair(m.m_data.values.data() + m.m_data.keys_host[key],
                              m.m_data.values.data() + m.m_data.keys_host[key + 1]);
      }
      auto equal_range(auto key) const {
        const auto& m = static_cast<const TMap&>(*this);
        return std::make_pair(m.m_data.values.data() + m.m_data.keys_host[key],
                              m.m_data.values.data() + m.m_data.keys_host[key + 1]);
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
