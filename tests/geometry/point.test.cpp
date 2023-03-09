#include <odrc/geometry/point.hpp>

#include <doctest/doctest.h>

#include <odrc/geometry/geometry.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry pointnd tests") {
  TEST_CASE("test getter/setter for integer coordinates") {
    odrc::geometry::pointnd p(0, 0);
    CHECK_EQ(p[0], 0);
    CHECK_EQ(p[1], 0);
    p[0] = 3;
    p[1] = 5;
    CHECK_EQ(p[0], 3);
    CHECK_EQ(p[1], 5);
    CHECK_EQ(p.at(0), 3);
    CHECK_EQ(p.at(1), 5);
    CHECK_EQ(p.x(), 3);
    CHECK_EQ(p.y(), 5);
    CHECK_THROWS_AS(p.z(), std::out_of_range);
    CHECK_THROWS_AS(p.m(), std::out_of_range);
    p.x(5);
    p.y(7);
    CHECK_EQ(p.x(), 5);
    CHECK_EQ(p.y(), 7);
  }
  TEST_CASE("test getter/setter for 5d integer coordinates") {
    odrc::geometry::pointnd<odrc::geometry::geometry_space<int, 5>> p;
    p[0] = 3;
    p[1] = 5;
    p[2] = 7;
    p[3] = 11;
    p[4] = 13;
    CHECK_EQ(p[0], 3);
    CHECK_EQ(p[1], 5);
    CHECK_EQ(p[2], 7);
    CHECK_EQ(p[3], 11);
    CHECK_EQ(p[4], 13);
    CHECK_THROWS_AS(p.at(666), std::out_of_range);
  }
  TEST_CASE("test getter/setter for double coordinates") {
    odrc::geometry::pointnd<odrc::geometry::geometry_space<double>> p(0.3, 0.5);
    CHECK_EQ(p[0], doctest::Approx(double{0.3}));
    CHECK_EQ(p[1], doctest::Approx(double{0.5}));
    p[0] = 3.5;
    p[1] = 5.7;
    CHECK_EQ(p[0], doctest::Approx(double{3.5}));
    CHECK_EQ(p[1], doctest::Approx(double{5.7}));
  }
  TEST_CASE("test initializer list constructor") {
    odrc::geometry::pointnd p{{1, 2}};
    CHECK_EQ(p[0], 1);
    CHECK_EQ(p[1], 2);
    odrc::geometry::pointnd<odrc::geometry::geometry_space<int, 3>> p3d{
        {1, 2, 3}};
    CHECK_EQ(p3d[0], 1);
    CHECK_EQ(p3d[1], 2);
    CHECK_EQ(p3d[2], 3);
  }
  TEST_CASE("test comparison operators") {
    odrc::geometry::pointnd p1{1, 1};
    odrc::geometry::pointnd p2{2, 2};
    CHECK_LT(p1, p2);
    CHECK_LE(p1, p2);
    CHECK_GT(p2, p1);
    CHECK_GE(p2, p2);
    CHECK_NE(p1, p2);
    CHECK_FALSE(p1 == p2);
    p2[0] = 1;
    CHECK_FALSE(p1 < p2);
    p2[1] = 1;
    CHECK_EQ(p1, p2);
    CHECK_FALSE(p1 < p2);
  }
  TEST_CASE("test non-comparable pointnds") {
    odrc::geometry::pointnd p1(1, 2);
    odrc::geometry::pointnd p2(2, 1);
    CHECK_FALSE(p1 < p2);
    CHECK_FALSE(p1 > p2);
    CHECK_FALSE(p1 <= p2);
    CHECK_FALSE(p1 >= p2);
    CHECK_FALSE(p1 == p2);
    CHECK_NE(p1, p2);
  }
}

TEST_SUITE("[OpenDRC] odrc::geometry point tests") {
  TEST_CASE("test point comparison") {
    odrc::geometry::point p1(odrc::geometry::pointnd<>{1, 1});
    odrc::geometry::point p2(odrc::geometry::pointnd<>{2, 2});
    CHECK_LT(p1, p2);
    CHECK_LE(p1, p2);
    CHECK_GT(p2, p1);
    CHECK_GE(p2, p2);
    CHECK_NE(p1, p2);
    CHECK_FALSE(p1 == p2);
    odrc::geometry::point p3(odrc::geometry::pointnd<>{1, 2});
    CHECK_FALSE(p1 < p3);
    CHECK_FALSE(p1 > p3);
    CHECK_FALSE(p1 == p3);
    CHECK_LE(p1, p3);
    CHECK_GE(p3, p1);
  }
  TEST_CASE("test non-comparable points") {
    odrc::geometry::point p1(odrc::geometry::pointnd<>{1, 2});
    odrc::geometry::point p2(odrc::geometry::pointnd<>{2, 1});
    CHECK_FALSE(p1 < p2);
    CHECK_FALSE(p1 > p2);
    CHECK_FALSE(p1 <= p2);
    CHECK_FALSE(p1 >= p2);
    CHECK_FALSE(p1 == p2);
    CHECK_NE(p1, p2);
  }
}