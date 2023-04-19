#include <odrc/geometry/polygon.hpp>

#include <list>
#include <vector>

#include <doctest/doctest.h>

#include <odrc/geometry/point.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry polygon tests") {
  using poly_t  = odrc::geometry::polygon<>;
  using point_t = odrc::geometry::point<>;
  TEST_CASE("test empty polygon") {
    poly_t poly;
    CHECK_EQ(poly.size(), 0);
  }
  TEST_CASE("test polygon construction with point list") {
    std::vector<point_t> points{{1, 1}, {1, 3}, {3, 3}, {3, 1}};
    poly_t               poly(points);
    CHECK_EQ(poly.size(), points.size());
  }
  TEST_CASE("test polygon construction with iterators") {
    std::list<point_t> points{{1, 1}, {1, 3}, {3, 3}, {3, 1}};
    poly_t             poly(points.begin(), points.end());
    CHECK_EQ(poly.size(), points.size());
  }
  TEST_CASE("test polygon construction with initializer list") {
    poly_t poly{{1, 1}, {1, 3}, {3, 3}, {3, 1}};
    CHECK_EQ(poly.size(), 4);
  }
  TEST_CASE("test polygon clear") {
    poly_t poly{{1, 1}, {1, 3}, {3, 3}, {3, 1}};
    CHECK_EQ(poly.size(), 4);
    poly.clear();
    CHECK_EQ(poly.size(), 0);
  }
  TEST_CASE("test polygon translation") {
    poly_t poly{{1, 1}, {1, 3}, {3, 3}, {3, 1}};
    poly_t poly2 = poly + point_t{1, 1};
    CHECK_EQ(poly2[0], point_t{2, 2});
    CHECK_EQ(poly2[1], point_t{2, 4});
    CHECK_EQ(poly2[2], point_t{4, 4});
    CHECK_EQ(poly2[3], point_t{4, 2});
  }
}