

#pragma once

#include "xstl/internal/map_interface.hpp"
#include "xstl/alpaka/detail/defines.hpp"
#include <alpaka/alpaka.hpp>
#include <optional>
#include <utility>

namespace xstd {
  namespace alpaka {

    template <typename T>
    association_map<T>::iterator association_map<T>::find_impl(key_type key) {
      return {m_data.values.data() + m_data.keys_host[key]};
    }

    template <typename T>
    association_map<T>::const_iterator association_map<T>::find_impl(key_type key) const {
      return {m_data.values.data() + m_data.keys_host[key]};
    }

    template <typename T>
    association_map<T>::size_type association_map<T>::count_impl(key_type key) const {
      if (key < 0 || static_cast<size_type>(key) >= m_keys)
        throw std::out_of_range("Key is out of range.");
      return m_data.keys_host[key + 1] - m_data.keys_host[key];
    }

    template <typename T>
    association_map<T>::bool association_map<T>::contains_impl(key_type key) const {
      if (key < 0 || static_cast<size_type>(key) >= m_keys)
        throw std::out_of_range("Key is out of range.");
      return m_data.keys_host[key + 1] > m_data.keys_host[key];
    }

    template <typename T>
    association_map<T>::iterator association_map<T>::lower_bound_impl(key_type key) {
      return {m_data.values.data() + m_data.keys_host[key]};
    }
    template <typename T>
    association_map<T>::const_iterator association_map<T>::lower_bound_impl(key_type key) const {
      return {m_data.values.data() + m_data.keys_host[key]};
    }

    template <typename T>
    association_map<T>::iterator association_map<T>::upper_bound_impl(key_type key) {
      return {m_data.values.data() + m_data.keys_host[key + 1]};
    }
    template <typename T>
    association_map<T>::const_iterator association_map<T>::upper_bound_impl(key_type key) const {
      return {m_data.values.data() + m_data.keys_host[key + 1]};
    }

    template <typename T>
    std::pair<association_map<T>::iterator, association_map<T>::iterator>
    association_map<T>::equal_range_impl(key_type key) {
      return std::make_pair(m_data.values.data() + m_data.keys_host[key],
                            m_data.values.data() + m_data.keys_host[key + 1]);
    }
    template <typename T>
    std::pair<association_map<T>::const_iterator, association_map<T>::const_iterator>
    association_map<T>::equal_range_impl(key_type key) const {
      return std::make_pair(m_data.values.data() + m_data.keys_host[key],
                            m_data.values.data() + m_data.keys_host[key + 1]);
    }

    template <typename T>
    void association_map<T>::fill_impl(std::span<association_map<T>::key_type> keys,
                                       std::span<association_map<T>::mapped_type> values) {}

    template <typename T>
    association_map<T>::View view_impl() {
      return m_view;
    }

  }  // namespace alpaka
}  // namespace xstd
