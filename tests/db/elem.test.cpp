#include <odrc/db/elem.hpp>

#include <doctest/doctest.h>

#include <odrc/geometry/point.hpp>
#include <odrc/geometry/polygon.hpp>

TEST_SUITE("[OpenDRC] odrc::db element tests") {
  TEST_CASE("test empty element") {
    odrc::db::element empty_elem;
    CHECK_EQ(empty_elem.get_polygon().size(), 0);
  }
  TEST_CASE("test element with l-value polygon") {
    odrc::geometry::polygon poly{{0, 0}, {0, 3}, {3, 3}, {3, 0}};
    odrc::db::element       elem(poly);
    CHECK_EQ(elem.get_polygon().size(), 4);
  }
  TEST_CASE("test element with r-value polygon") {
    using polygon = odrc::geometry::polygon<>;
    odrc::db::element elem(polygon{{0, 0}, {0, 3}, {3, 3}, {3, 0}});
    CHECK_EQ(elem.get_polygon().size(), 4);
  }
  TEST_CASE("test element with customized polygon type") {
    using point3d = odrc::geometry::point<int, 3>;
    using polygon = odrc::geometry::polygon<point3d>;
    odrc::db::element elem(polygon{{0, 0, 0}, {0, 3, 0}, {3, 3, 0}, {3, 0, 0}});
    CHECK_EQ(elem.get_polygon().size(), 4);
  }
}