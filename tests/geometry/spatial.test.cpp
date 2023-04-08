#include <odrc/geometry/spatial.hpp>

#include <doctest/doctest.h>

#include <odrc/geometry/box.hpp>

TEST_SUITE("[OpenDRC] odrc::geometry spatial tests") {
  using point = odrc::geometry::point<>;
  using box   = odrc::geometry::box<>;
  TEST_CASE("test spatial vector constructors") {
    box b0{point{0, 0}, {2, 2}};
    box b1{point{3, 0}, {5, 2}};
    box b2{point{6, 0}, {8, 2}};

    odrc::geometry::spatial_vector<> sv1{b0, b1, b2};
    CHECK_EQ(sv1.size(), 3);

    std::vector<box>                 objs{b0, b1, b2};
    odrc::geometry::spatial_vector<> sv2{objs};
    CHECK_EQ(sv2.size(), 3);
    odrc::geometry::spatial_vector<> sv3{objs.begin(), objs.end()};
    CHECK_EQ(sv3.size(), 3);
  }
  TEST_CASE("test spatial vector queries") {
    box b0{point{0, 0}, {2, 2}};
    box b1{point{3, 0}, {5, 2}};
    box b2{point{6, 0}, {8, 2}};

    odrc::geometry::spatial_vector<> sv{b0, b1, b2};

    std::vector<box> ovlp;
    sv.query(box{point{1, 0}, {4, 2}}, std::back_inserter(ovlp));
    CHECK_EQ(ovlp.size(), 2);
    CHECK_EQ(ovlp.at(0), b0);
    CHECK_EQ(ovlp.at(1), b1);
  }
}
