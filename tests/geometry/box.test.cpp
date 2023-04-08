#include <odrc/geometry/box.hpp>

#include <doctest/doctest.h>

#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry box tests") {
  using odrc::geometry::box;
  TEST_CASE("test 2d box") {
    using gs      = odrc::geometry::geometry_space<int, 2>;
    using point_t = odrc::geometry::point<gs>;
    point_t minp{0, 0};
    point_t maxp{3, 3};
    box     b(minp, maxp);
    CHECK_EQ(b.min_corner().x(), 0);
    CHECK_EQ(b.min_corner().y(), 0);
    CHECK_EQ(b.max_corner().x(), 3);
    CHECK_EQ(b.max_corner().y(), 3);
    box b2(minp, maxp);
    CHECK_EQ(b, b2);
    box b3(minp, {5, 5});
    CHECK_FALSE(b == b3);
  }

  TEST_CASE("test 3d box") {
    using gs      = odrc::geometry::geometry_space<int, 3>;
    using point_t = odrc::geometry::point<gs>;
    point_t minp{0, 0, 0};
    point_t maxp{3, 3, 3};
    box     b(minp, maxp);
    CHECK_EQ(b.min_corner().x(), 0);
    CHECK_EQ(b.min_corner().y(), 0);
    CHECK_EQ(b.min_corner().z(), 0);
    CHECK_EQ(b.max_corner().x(), 3);
    CHECK_EQ(b.max_corner().y(), 3);
    CHECK_EQ(b.max_corner().z(), 3);
    box b2(minp, maxp);
    CHECK_EQ(b, b2);
    box b3(minp, {5, 5, 5});
    CHECK_FALSE(b == b3);
  }

  TEST_CASE("test 1d overlap query") {
    using gs      = odrc::geometry::geometry_space<int, 1>;
    using point_t = odrc::geometry::point<gs>;
    box b1(point_t{0}, point_t{2});
    box b2(point_t{3}, point_t{4});
    CHECK_FALSE(b1.overlaps(b2));
    b2.min_corner().x() = 2;
    CHECK_FALSE(b1.overlaps(b2));  // touching is not overlapping
    b2.min_corner().x() = 1;
    CHECK(b1.overlaps(b2));
    CHECK(b2.overlaps(b1));
  }

  TEST_CASE("test 2d overlap query") {
    using gs      = odrc::geometry::geometry_space<>;
    using point_t = odrc::geometry::point<gs>;
    box b1(point_t{0, 0}, point_t{2, 2});
    box b2(point_t{3, 3}, point_t{4, 4});
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