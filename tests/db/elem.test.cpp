#include <odrc/db/elem.hpp>

#include <doctest/doctest.h>

#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>
#include <odrc/geometry/polygon.hpp>

TEST_SUITE("[OpenDRC] odrc::db element tests") {
  TEST_CASE("test empty element") {
    odrc::db::element empty_elem;
    CHECK_EQ(empty_elem.get_polygon().size(), 0);
  }
  TEST_CASE("test element with l-value polygon") {
    odrc::geometry::polygon poly{{0, 0}, {0, 3}, {3, 3}, {3, 0}};
    odrc::db::element       elem(1, poly);
    CHECK_EQ(elem.get_layer(), 1);
    CHECK_EQ(elem.get_polygon().size(), 4);
  }
  TEST_CASE("test element with r-value polygon") {
    using polygon = odrc::geometry::polygon<>;
    odrc::db::element elem(1, polygon{{0, 0}, {0, 3}, {3, 3}, {3, 0}});
    CHECK_EQ(elem.get_layer(), 1);
    CHECK_EQ(elem.get_polygon().size(), 4);
  }
  TEST_CASE("test element with customized polygon type") {
    using gs      = odrc::geometry::geometry_space<int, 3>;
    using polygon = odrc::geometry::polygon<gs>;
    odrc::db::element elem(1,
                           polygon{{0, 0, 0}, {0, 3, 0}, {3, 3, 0}, {3, 0, 0}});
    CHECK_EQ(elem.get_layer(), 1);
    CHECK_EQ(elem.get_polygon().size(), 4);
  }
}