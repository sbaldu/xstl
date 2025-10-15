

#pragma once

#include "xstl/cpu/association_map.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <utility>

namespace xstd {

  template <typename T>
  association_map<T>::Extents association_map<T>::extents_impl() const {
    return {m_values, m_keys};
  }

  template <typename T>
  void association_map<T>::fill_impl(std::span<association_map<T>::key_type> keys,
                                     std::span<association_map<T>::mapped_type> values) {
    std::vector<key_type> accumulator(m_keys, 0);
    std::for_each(keys.begin(), keys.end(), [&](key_type key) { accumulator[key]++; });
    std::vector<key_type> temporary_keys(m_keys + 1);
    temporary_keys[0] = 0;
    std::inclusive_scan(accumulator.begin(), accumulator.end(), temporary_keys.begin() + 1);
    std::copy(temporary_keys.begin(), temporary_keys.end(), m_data.keys.begin());
    for (auto i = 0u; i < keys.size(); ++i) {
      auto& offset = temporary_keys[keys[i]];
      m_data.values[offset] = values[i];
      ++offset;
    }
  }

}  // namespace xstd
