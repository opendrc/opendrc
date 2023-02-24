#include <odrc/geometry/point.hpp>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc::geometry point tests") {
  TEST_CASE("test getter/setter for integer coordinates") {
    odrc::geometry::point p(0, 0);
    CHECK_EQ(p.get<0>(), 0);
    CHECK_EQ(p.get<1>(), 0);
    p.set<0>(3);
    p.set<1>(5);
    CHECK_EQ(p.get<0>(), 3);
    CHECK_EQ(p.get<1>(), 5);
  }
  TEST_CASE("test getter/setter for 5d integer coordinates") {
    odrc::geometry::point<int, 5> p;
    p.set<0>(3);
    p.set<1>(5);
    p.set<2>(7);
    p.set<3>(11);
    p.set<4>(13);
    CHECK_EQ(p.get<0>(), 3);
    CHECK_EQ(p.get<1>(), 5);
    CHECK_EQ(p.get<2>(), 7);
    CHECK_EQ(p.get<3>(), 11);
    CHECK_EQ(p.get<4>(), 13);
  }
  TEST_CASE("test getter/setter for double coordinates") {
    odrc::geometry::point<double> p(0.3, 0.5);
    CHECK_EQ(p.get<0>(), doctest::Approx(double{0.3}));
    CHECK_EQ(p.get<1>(), doctest::Approx(double{0.5}));
    p.set<0>(3.5);
    p.set<1>(5.7);
    CHECK_EQ(p.get<0>(), doctest::Approx(double{3.5}));
    CHECK_EQ(p.get<1>(), doctest::Approx(double{5.7}));
  }
  TEST_CASE("test x/y getter/setter for point2d") {
    odrc::geometry::point2d p;
    p.x(3);
    p.y(5);
    CHECK_EQ(p.x(), 3);
    CHECK_EQ(p.y(), 5);
  }
}