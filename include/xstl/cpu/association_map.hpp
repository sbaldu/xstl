/// @file association_map.hpp
/// @brief A header file defining the association_map class, which is a map-like structure
/// @author Simone Balducci

#pragma once

#include "xstl/internal/map_interface.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <utility>

namespace xstd {

  /// association_map is a map-like structure that associates keys with values.
  /// It is designed to efficiently store and retrieve values associated with integer keys.
  ///
  /// @tparam T The type of the values stored in the association map.
  template <typename T>
  class association_map : public internal::map_interface<association_map<T>> {
  public:
    using key_type = int32_t;
    using mapped_type = T;
    using value_type = T;
    using size_type = std::size_t;
    using iterator = value_type*;
    using const_iterator = const value_type*;
    using key_container_type = std::vector<key_type>;
    using mapped_container_type = std::vector<mapped_type>;

    struct containers {
      mapped_container_type values;
      key_container_type keys;
      key_container_type& keys_host;

      explicit containers(size_type values_size, size_type keys_size)
          : values(values_size), keys(keys_size + 1), keys_host{keys} {}
    };

    struct Extents {
      size_type values;
      size_type keys;
    };

    struct View {
      value_type* m_values;
      key_type* m_keys;
      Extents m_extents;

      constexpr auto operator[](key_type key) const {
        const auto offset = m_keys[key];
        const auto size = m_keys[key + 1] - offset;
        return std::span<const value_type>(m_values + offset, size);
      }

      constexpr auto operator[](key_type key) {
        const auto offset = m_keys[key];
        const auto size = m_keys[key + 1] - offset;
        return std::span<value_type>(m_values + offset, size);
      }
      constexpr auto extents() const { return m_extents; }
    };

    /// @brief Constructs an association map with a specified number of values and keys.
    ///
    /// @param values The number of values to be stored in the association map.
    /// @param keys The number of bins (or keys) in the association map.
    explicit association_map(size_type values, size_type keys)
        : m_data(values, keys),
          m_view{m_data.values.data(), m_data.keys.data(), Extents{values, keys}},
          m_values{values},
          m_keys{keys} {}

#ifdef XSTL_BUILD_DOXYGEN
    /// @brief Checks if the association map is empty.
    ///
    /// @return True if the association map is empty, false otherwise.
    auto empty() const;
    /// @brief Returns the number of values in the association map.
    ///
    /// @return The number of values in the association map.
    auto size() const;
    /// @brief Returns the extents of the containers in the association map.
    ///
    /// @return An Extents object containing the number of keys and values.
    auto extents() const;

    /// @brief Returns an iterator to the beginning of the values array.
    ///
    /// @return An iterator to the first element in the values array.
    iterator begin();
    /// @brief Returns a const iterator to the beginning of the values array.
    ///
    /// @return A const iterator to the first element in the values array.
    const_iterator begin() const;
    /// @brief Returns a const iterator to the beginning of the values array.
    ///
    /// @return A const iterator to the first element in the values array.
    const_iterator cbegin() const;

    /// @brief Returns an iterator to the end of the values array.
    ///
    /// @return An iterator to one past the last element in the values array.
    iterator end();
    const_iterator end() const;
    /// @brief Returns a const iterator to the end of the values array.
    ///
    /// @return A const iterator to one past the last element in the values array.
    const_iterator cend() const;

    /// @brief Returns an iterator to the first element with a specific key.
    ///
    /// @param key The key to search for.
    /// @return An iterator to the first element with the specified key, or end() if not found.
    iterator find(key_type key);
    /// @brief Returns a const iterator to the first element with a specific key.
    ///
    /// @param key The key to search for.
    /// @return A const iterator to the first element with the specified key, or end() if not found.
    const_iterator find(key_type key) const;

    /// @brief Returns the number of values with a specific key.
    ///
    /// @param key The key to count.
    /// @return The number of values with the specified key.
    size_type count(key_type key) const;

    /// @brief Checks if the association map contains values associated to a specific key.
    ///
    /// @param key The key to check for.
    /// @return True if the association map contains values for the specified key, false otherwise.
    bool contains(key_type key) const;

    /// @brief Returns an iterator to the first element with a key not less than the specified key.
    ///
    /// @param key The key to search for.
    /// @return An iterator to the first element with a key not less than the specified key.
    iterator lower_bound(key_type key);
    /// @brief Returns a const iterator to the first element with a key not less than the specified key.
    ///
    /// @param key The key to search for.
    /// @return A const iterator to the first element with a key not less than the specified key.
    const_iterator lower_bound(key_type key) const;

    /// @brief Returns an iterator to the first element with a key greater than the specified key.
    ///
    /// @param key The key to search for.
    /// @return An iterator to the first element with a key greater than the specified key.
    iterator upper_bound(key_type key);
    /// @brief Returns a const iterator to the first element with a key greater than the specified key.
    ///
    /// @param key The key to search for.
    /// @return A const iterator to the first element with a key greater than the specified key.
    const_iterator upper_bound(key_type key) const;

    /// @brief Returns a pair of iterators representing the range of values with a specific key.
    ///
    /// @param key The key to search for.
    /// @return A pair of iterators representing the range of values with the specified key.
    std::pair<iterator, iterator> equal_range(key_type key);
    /// @brief Returns a pair of const iterators representing the range of values with a specific key.
    ///
    /// @param key The key to search for.
    /// @return A pair of const iterators representing the range of values with the specified key.
    std::pair<const_iterator, const_iterator> equal_range(key_type key) const;

    /// @brief Fills the association map with keys and values from the provided spans.
    ///
    /// @param keys A span of keys to be associated with the values.
    /// @param values A span of values to be associated with the keys.
    void fill(std::span<key_type> keys, std::span<mapped_type> values);

    /// @brief Returns a view of the association map.
    ///
    /// @return A pointer to a View of the association map.
    View view();
#endif

  private:
    containers m_data;
    View m_view;
    size_type m_values;
    size_type m_keys;

  private:
    Extents extents_impl() const;

    void fill_impl(std::span<key_type> keys, std::span<mapped_type> values);

    friend struct internal::map_interface<association_map<T>>;
  };

}  // namespace xstd

#include "xstl/cpu/detail/association_map.hpp"
