#include <odrc/geometry/box.hpp>

#include <doctest/doctest.h>

#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry boxnd tests") {
  using odrc::geometry::boxnd;
  TEST_CASE("test 2d boxnd") {
    using gs      = odrc::geometry::geometry_space<int, 2>;
    using p2d     = odrc::geometry::pointnd<gs>;
    using point_t = odrc::geometry::point<gs>;
    point_t minp(p2d{0, 0});
    point_t maxp(p2d{3, 3});
    boxnd   b(minp, maxp);
    CHECK_EQ(b.min_corner().x(), 0);
    CHECK_EQ(b.min_corner().y(), 0);
    CHECK_EQ(b.max_corner().x(), 3);
    CHECK_EQ(b.max_corner().y(), 3);
  }

  TEST_CASE("test 3d boxnd") {
    using gs      = odrc::geometry::geometry_space<int, 3>;
    using p3d     = odrc::geometry::pointnd<gs>;
    using point_t = odrc::geometry::point<gs>;
    point_t minp{p3d{0, 0, 0}};
    point_t maxp{p3d{3, 3, 3}};
    boxnd   b(minp, maxp);
    CHECK_EQ(b.min_corner().x(), 0);
    CHECK_EQ(b.min_corner().y(), 0);
    CHECK_EQ(b.min_corner().z(), 0);
    CHECK_EQ(b.max_corner().x(), 3);
    CHECK_EQ(b.max_corner().y(), 3);
    CHECK_EQ(b.max_corner().z(), 3);
  }

  TEST_CASE("test 1d overlap query") {
    using gs      = odrc::geometry::geometry_space<int, 1>;
    using p1d     = odrc::geometry::pointnd<gs>;
    using point_t = odrc::geometry::point<gs>;
    boxnd b1(point_t{p1d{0}}, point_t{p1d{2}});
    boxnd b2(point_t{p1d{3}}, point_t{p1d{4}});
    CHECK_FALSE(b1.overlaps(b2));
    b2.min_corner().x() = 2;
    CHECK_FALSE(b1.overlaps(b2));  // touching is not overlapping
    b2.min_corner().x() = 1;
    CHECK(b1.overlaps(b2));
    CHECK(b2.overlaps(b1));
  }

  TEST_CASE("test 2d overlap query") {
    using gs      = odrc::geometry::geometry_space<>;
    using p2d     = odrc::geometry::pointnd<gs>;
    using point_t = odrc::geometry::point<gs>;
    boxnd b1(point_t{p2d{0, 0}}, point_t{p2d{2, 2}});
    boxnd b2(point_t{p2d{3, 3}}, point_t{p2d{4, 4}});
    CHECK_FALSE(b1.overlaps(b2));
    b2.min_corner().x() = 2;
    b2.min_corner().y() = 2;
    CHECK_FALSE(b1.overlaps(b2));  // corner touching is not overlapping
    b2.min_corner().x() = 0;
    CHECK_FALSE(b1.overlaps(b2));  // edge touching is not overlapping
    b2.min_corner().x() = 1;
    b2.min_corner().y() = 1;
    CHECK(b1.overlaps(b2));
    CHECK(b2.overlaps(b1));
  }
}