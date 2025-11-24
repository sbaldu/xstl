
#include "xstl/xstl.hpp"
#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using namespace xstd::alpaka;

TEST_CASE("Test a simple binary association map") {
  auto device = alpaka::getDevByIdx(Platform{}, 0u);
  Queue queue(device);
  xstd::alpaka::association_map<int> map(5, 2, queue);
  auto keys = alpaka::allocBuf<int, internal::Idx>(device, internal::Vec1D{5});
  auto values = alpaka::allocBuf<int, internal::Idx>(device, internal::Vec1D{5});
  keys[0] = 0;
  keys[1] = 1;
  keys[2] = 0;
  keys[3] = 1;
  keys[4] = 0;
  values[0] = 0;
  values[1] = 1;
  values[2] = 2;
  values[3] = 3;
  values[4] = 4;
  map.fill(queue, std::span<int>(keys.data(), 5), std::span<int>(values.data(), 5));

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
    xstd::alpaka::association_map<int> empty_map(0, 0, queue);
    CHECK(empty_map.empty());
  }
  SUBCASE("Test the find method") {
    CHECK(map.find(0) == map.begin());
    CHECK(map.find(1) == map.begin() + 3);
    CHECK(*map.find(0) == 0);
    CHECK(*map.find(1) == 1);
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
  SUBCASE("Test the view method") {
    auto view = map.view();
    CHECK(view[0].size() == 3);
    CHECK(view[1].size() == 2);
    CHECK(view[0][0] == 0);
    CHECK(view[0][1] == 2);
    CHECK(view[0][2] == 4);
    CHECK(view[1][0] == 1);
    CHECK(view[1][1] == 3);
  }
}

TEST_CASE("Test binary association map with floats") {
  auto device = alpaka::getDevByIdx(Platform{}, 0u);
  Queue queue(device);
  xstd::alpaka::association_map<float> map(5, 2, queue);
  auto keys = alpaka::allocBuf<int, internal::Idx>(device, internal::Vec1D{5});
  auto values = alpaka::allocBuf<float, internal::Idx>(device, internal::Vec1D{5});
  keys[0] = 0;
  keys[1] = 1;
  keys[2] = 0;
  keys[3] = 1;
  keys[4] = 0;
  values[0] = 0.f;
  values[1] = 1.f;
  values[2] = 2.f;
  values[3] = 3.f;
  values[4] = 4.f;
  map.fill(queue, std::span<int>(keys.data(), 5), std::span<float>(values.data(), 5));

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
    xstd::alpaka::association_map<int> empty_map(0, 0, queue);
    CHECK(empty_map.empty());
  }
  SUBCASE("Test the find method") {
    CHECK(map.find(0) == map.begin());
    CHECK(map.find(1) == map.begin() + 3);
    CHECK(*map.find(0) == 0);
    CHECK(*map.find(1) == 1);
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
  SUBCASE("Test the view method") {
    auto view = map.view();
    CHECK(view[0].size() == 3);
    CHECK(view[1].size() == 2);
    CHECK(view[0][0] == 0.f);
    CHECK(view[0][1] == 2.f);
    CHECK(view[0][2] == 4.f);
    CHECK(view[1][0] == 1.f);
    CHECK(view[1][1] == 3.f);
  }
}
