#include <odrc/geometry/polygon.hpp>

#include <list>
#include <vector>

#include <doctest/doctest.h>

#include <odrc/geometry/point.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry polygon tests") {
  using poly_t  = odrc::geometry::detail::polygon<>;
  using point_t = odrc::geometry::point<>;
  using p2d     = odrc::geometry::pointnd<>;
  TEST_CASE("test empty polygon") {
    odrc::geometry::polygon poly{poly_t{}};
    CHECK_EQ(poly.size(), 0);
  }
  TEST_CASE("test polygon with point list") {
    std::vector<point_t>    points{p2d{1, 1}, p2d{1, 3}, p2d{3, 3}, p2d{3, 1}};
    odrc::geometry::polygon poly{poly_t(points)};
    CHECK_EQ(poly.size(), points.size());
  }
  TEST_CASE("test polygon with iterators") {
    std::list<point_t>      points{p2d{1, 1}, p2d{1, 3}, p2d{3, 3}, p2d{3, 1}};
    odrc::geometry::polygon poly(poly_t(points.begin(), points.end()));
    CHECK_EQ(poly.size(), points.size());
  }
  TEST_CASE("test polygon with initializer list") {
    odrc::geometry::polygon poly(
        poly_t({p2d{1, 1}, p2d{1, 3}, p2d{3, 3}, p2d{3, 1}}));
    CHECK_EQ(poly.size(), 4);
  }
}