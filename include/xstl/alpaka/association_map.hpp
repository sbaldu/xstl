

#pragma once

#include "xstl/internal/map_interface.hpp"
#include "xstl/alpaka/detail/defines.hpp"
#include <alpaka/alpaka.hpp>
#include <optional>
#include <utility>

namespace xstd {
  namespace alpaka {

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
      using key_container_type = alpaka::Buf<Device, key_type, Dim1D, Idx>;
      using mapped_container_type = alpaka::Buf<Device, mapped_type, Dim1D, Idx>;
	  using key_container_host_type = alpaka::Buf<alpaka::DevCpu, key_type, Dim1D, Idx>;

      struct containers {
        mapped_container_type values;
        key_container_type keys;
		std::optional<key_container_host_type> keys_host;

        template <typename TQueue>
          requires std::same<Device, alpaka::DevCpu>
        explicit containers(key_type values_size, key_type keys_size, const TQueue&) {
          values = alpaka::allocMappedBuf<mapped_type, Idx>(host, Platform(), Vec1D{values_size}));
          keys = alpaka::allocMappedBuf<key_type, Idx>(host, Platform(), Vec1D{keys_size + 1});
        }
        template <typename TQueue>
          requires(not std::same<Device, alpaka::DevCpu>)
        explicit containers(key_type values_size, key_type keys_size, const TQueue& queue) {
          values = alpaka::allocAsyncBuf<mapped_type, Idx>(queue, Vec1D{values_size}));
          keys = alpaka::allocAsyncBuf<key_type, Idx>(queue, Vec1D{keys_size + 1});
		  keys_host = alpaka::allocMappedBuf<key_type, Idx>(host, Platform(), Vec1D{keys_size + 1});
        }
      };

      struct Extents {
        size_type values;
        size_type keys;
      };

      struct View {
        value_type* values;
        key_type* keys;

        ALPAKA_FN_ACC constexpr std::span<value_type> operator[](key_type key) const {
          const auto offset = keys[key];
          const auto size = keys[key + 1] - offset;
          return std::span<const value_type>(values + offset, size);
        }

        ALPAKA_FN_ACC constexpr std::span<value_type> operator[](key_type key) {
          const auto offset = keys[key];
          const auto size = keys[key + 1] - offset;
          return std::span<value_type>(values + offset, size);
        }
      };

      /// @brief Constructs an association map with a specified number of elements and keys.
      ///
      /// @param nelements The number of elements to be stored in the association map.
      /// @param nkeys The number of bins (or keys) in the association map.
	  template <typename TQueue>
      explicit association_map(size_type elements, size_type keys, const TQueue& queue)
          : m_data(elements, keys, queue),
            m_view{m_data.values.data(), m_data.keys.data()},
            m_values{elements},
            m_keys{keys} {}

#ifdef XSTL_BUILD_DOXYGEN
      /// @brief Checks if the association map is empty.
      ///
      /// @return True if the association map is empty, false otherwise.
      auto empty() const;
      /// @brief Returns the number of elements in the association map.
      ///
      /// @return The number of elements in the association map.
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

      /// @brief Returns the number of elements with a specific key.
      ///
      /// @param key The key to count.
      /// @return The number of elements with the specified key.
      size_type count(key_type key) const;

      /// @brief Checks if the association map contains elements associated to a specific key.
      ///
      /// @param key The key to check for.
      /// @return True if the association map contains elements for the specified key, false otherwise.
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

      /// @brief Returns a pair of iterators representing the range of elements with a specific key.
      ///
      /// @param key The key to search for.
      /// @return A pair of iterators representing the range of elements with the specified key.
      std::pair<iterator, iterator> equal_range(key_type key);
      /// @brief Returns a pair of const iterators representing the range of elements with a specific key.
      ///
      /// @param key The key to search for.
      /// @return A pair of const iterators representing the range of elements with the specified key.
      std::pair<const_iterator, const_iterator> equal_range(key_type key) const;

      /// @brief Fills the association map with keys and values from the provided spans.
      ///
      /// @param keys A span of keys to be associated with the values.
      /// @param values A span of values to be associated with the keys.
      void fill(std::span<key_type> keys, std::span<mapped_type> values);

      /// @brief Returns a view of the association map.
      ///
      /// @return A pointer to a View of the association map.
      View view() { return m_view; }
#endif

    private:
      containers m_data;
      View m_view;
      size_type m_values;
      size_type m_keys;

    private:
      Extents extents_impl() const;

      iterator find_impl(key_type key);
      const_iterator find_impl(key_type key) const;

      size_type count_impl(key_type key) const;

      bool contains_impl(key_type key) const;

      iterator lower_bound_impl(key_type key);
      const_iterator lower_bound_impl(key_type key) const;

      iterator upper_bound_impl(key_type key);
      const_iterator upper_bound_impl(key_type key) const;

      std::pair<iterator, iterator> equal_range_impl(key_type key);
      std::pair<const_iterator, const_iterator> equal_range_impl(key_type key) const;

      void fill_impl(std::span<key_type> keys, std::span<mapped_type> values);

      View* view_impl();

      friend struct internal::map_interface<association_map<T>>;
    };

  }  // namespace alpaka
}  // namespace xstd
