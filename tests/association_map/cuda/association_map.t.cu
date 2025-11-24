
#include "xstl/xstl.hpp"
#include <iostream>
#include <thrust/device_vector.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test a simple binary association map") {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  xstd::cuda::association_map<int> map(5, 2, stream);
  thrust::device_vector<int> keys{0, 1, 0, 1, 0};
  thrust::device_vector<int> values{0, 1, 2, 3, 4};
  map.fill(stream, std::span<int>(keys.data().get(), 5), std::span<int>(values.data().get(), 5));

  SUBCASE("Check size of map") { CHECK(map.size() == 5); }
  SUBCASE("Check extents of map's containers") {
    auto extents = map.extents();
    CHECK(extents.keys == 2);
    CHECK(extents.values == 5);
  }
  SUBCASE("Test the count method") {
    CHECK(map.count(0) == 3);
    CHECK(map.count(1) == 2);
  }
  SUBCASE("Test the contains method") {
    CHECK(map.contains(0));
    CHECK(map.contains(1));
    CHECK_THROWS(map.contains(2));
  }
  SUBCASE("Test the empty method") {
    CHECK(!map.empty());
    xstd::cuda::association_map<int> empty_map(0, 0, stream);
    CHECK(empty_map.empty());
  }
  SUBCASE("Test the find method") {
    CHECK(map.find(0) == map.begin());
    CHECK(map.find(1) == map.begin() + 3);
  }
  SUBCASE("Test the lower_bound and upper_bound methods") {
    auto lower = map.lower_bound(0);
    auto upper = map.upper_bound(0);
    CHECK(lower == map.begin());
    CHECK(upper == map.begin() + 3);

    lower = map.lower_bound(1);
    upper = map.upper_bound(1);
    CHECK(lower == map.begin() + 3);
    CHECK(upper == map.end());
  }
  SUBCASE("Test the equal_range method") {
    auto range = map.equal_range(0);
    CHECK(range.first == map.begin());
    CHECK(range.second == map.begin() + 3);

    range = map.equal_range(1);
    CHECK(range.first == map.begin() + 3);
    CHECK(range.second == map.end());
  }
  cudaStreamDestroy(stream);
}

TEST_CASE("Test binary association map with floats") {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  xstd::cuda::association_map<float> map(5, 2, stream);
  thrust::device_vector<int> keys{0, 1, 0, 1, 0};
  thrust::device_vector<float> values{0.f, 1.f, 2.f, 3.f, 4.f};
  map.fill(stream, std::span<int>(keys.data().get(), 5), std::span<float>(values.data().get(), 5));

  SUBCASE("Check size of map") { CHECK(map.size() == 5); }
  SUBCASE("Check extents of map's containers") {
    auto extents = map.extents();
    CHECK(extents.keys == 2);
    CHECK(extents.values == 5);
  }
  SUBCASE("Test the count method") {
    CHECK(map.count(0) == 3);
    CHECK(map.count(1) == 2);
  }
  SUBCASE("Test the contains method") {
    CHECK(map.contains(0));
    CHECK(map.contains(1));
    CHECK_THROWS(map.contains(2));
  }
  SUBCASE("Test the empty method") {
    CHECK(!map.empty());
    xstd::cuda::association_map<int> empty_map(0, 0, stream);
    CHECK(empty_map.empty());
  }
  SUBCASE("Test the find method") {
    CHECK(map.find(0) == map.begin());
    CHECK(map.find(1) == map.begin() + 3);
  }
  SUBCASE("Test the lower_bound and upper_bound methods") {
    auto lower = map.lower_bound(0);
    auto upper = map.upper_bound(0);
    CHECK(lower == map.begin());
    CHECK(upper == map.begin() + 3);

    lower = map.lower_bound(1);
    upper = map.upper_bound(1);
    CHECK(lower == map.begin() + 3);
    CHECK(upper == map.end());
  }
  SUBCASE("Test the equal_range method") {
    auto range = map.equal_range(0);
    CHECK(range.first == map.begin());
    CHECK(range.second == map.begin() + 3);

    range = map.equal_range(1);
    CHECK(range.first == map.begin() + 3);
    CHECK(range.second == map.end());
  }
  cudaStreamDestroy(stream);
}
