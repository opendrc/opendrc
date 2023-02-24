#include <odrc/geometry/polygon.hpp>

#include <list>
#include <vector>

#include <doctest/doctest.h>

#include <odrc/geometry/point.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry polygon tests") {
  using p = odrc::geometry::point2d<>;
  TEST_CASE("test empty polygon") {
    odrc::geometry::polygon<p> poly;
    CHECK_EQ(poly.size(), 0);
  }
  TEST_CASE("test polygon with point list") {
    std::vector<p>             points{{1, 1}, {1, 3}, {3, 3}, {3, 1}};
    odrc::geometry::polygon<p> poly(points);
    CHECK_EQ(poly.size(), points.size());
  }
  TEST_CASE("test polygon with iterators") {
    std::list<p>               points{{1, 1}, {1, 3}, {3, 3}, {3, 1}};
    odrc::geometry::polygon<p> poly(points.begin(), points.end());
    CHECK_EQ(poly.size(), points.size());
  }
  TEST_CASE("test polygon with initializer list") {
    odrc::geometry::polygon<p> poly({{1, 1}, {1, 3}, {3, 3}, {3, 1}});
    CHECK_EQ(poly.size(), 4);
  }
}