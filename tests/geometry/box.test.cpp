#include <odrc/geometry/box.hpp>

#include <doctest/doctest.h>

#include <odrc/geometry/point.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry box tests") {
  TEST_CASE("test box") {
    odrc::geometry::point2d minp(0, 0);
    odrc::geometry::point2d maxp(3, 3);
    odrc::geometry::box     b(minp, maxp);
    CHECK_EQ(b.min_corner().x(), 0);
    CHECK_EQ(b.min_corner().y(), 0);
    CHECK_EQ(b.max_corner().x(), 3);
    CHECK_EQ(b.max_corner().y(), 3);
  }

  TEST_CASE("test 3d box") {
    odrc::geometry::point<int, 3> minp{0, 0, 0};
    odrc::geometry::point<int, 3> maxp{3, 3, 3};
    odrc::geometry::box           b(minp, maxp);
    CHECK_EQ(b.min_corner().get<0>(), 0);
    CHECK_EQ(b.min_corner().get<1>(), 0);
    CHECK_EQ(b.min_corner().get<2>(), 0);
    CHECK_EQ(b.max_corner().get<0>(), 3);
    CHECK_EQ(b.max_corner().get<1>(), 3);
    CHECK_EQ(b.max_corner().get<2>(), 3);
  }

  TEST_CASE("test 1d overlap query") {
    using p1d = odrc::geometry::point<int, 1>;
    odrc::geometry::box b1(p1d{0}, p1d{2});
    odrc::geometry::box b2(p1d{3}, p1d{4});
    CHECK_FALSE(b1.overlaps(b2));
    b2.min_corner().set<0>(2);
    CHECK_FALSE(b1.overlaps(b2));  // touching is not overlapping
    b2.min_corner().set<0>(1);
    CHECK(b1.overlaps(b2));
    CHECK(b2.overlaps(b1));
  }

  TEST_CASE("test 2d overlap query") {
    using p2d = odrc::geometry::point2d<>;
    odrc::geometry::box b1(p2d{0, 0}, p2d{2, 2});
    odrc::geometry::box b2(p2d{3, 3}, p2d{4, 4});
    CHECK_FALSE(b1.overlaps(b2));
    b2.min_corner().set<0>(2);
    b2.min_corner().set<1>(2);
    CHECK_FALSE(b1.overlaps(b2));  // corner touching is not overlapping
    b2.min_corner().set<0>(0);
    CHECK_FALSE(b1.overlaps(b2));  // edge touching is not overlapping
    b2.min_corner().set<0>(1);
    b2.min_corner().set<1>(1);
    CHECK(b1.overlaps(b2));
    CHECK(b2.overlaps(b1));
  }
}