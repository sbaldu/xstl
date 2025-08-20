
#include "xstl/cpu/association_map.hpp"
#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Test a simple binary association map") {
  xstd::association_map<int> map(5, 2);
  std::vector<int> keys = {0, 1, 0, 1, 0};
  std::vector<int> values = {0, 1, 2, 3, 4};
  map.fill(keys, values);

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
    xstd::association_map<int> empty_map(0, 0);
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
    CHECK((*view)[0].size() == 3);
    CHECK((*view)[1].size() == 2);
    CHECK((*view)[0][0] == 0);
    CHECK((*view)[0][1] == 2);
    CHECK((*view)[0][2] == 4);
    CHECK((*view)[1][0] == 1);
    CHECK((*view)[1][1] == 3);
  }
}
