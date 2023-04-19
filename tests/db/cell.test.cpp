#include <odrc/db/cell.hpp>

#include <deque>
#include <list>

#include <doctest/doctest.h>

#include <odrc/db/elem.hpp>
#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>
#include <odrc/geometry/polygon.hpp>

TEST_SUITE("[OpenDRC] odrc::db cell tests") {
  TEST_CASE("test empty cell") {
    odrc::db::cell c(0, "empty_cell");
    CHECK_EQ(c.get_id(), 0);
    CHECK_EQ(c.get_name(), "empty_cell");
    CHECK_EQ(c.num_elements(), 0);
    CHECK_EQ(c.num_cell_refs(), 0);
  }
  TEST_CASE("test add element") {
    odrc::db::cell    c;
    odrc::db::element e1;
    e1.set_layer(1);
    c.insert(e1);
    CHECK_EQ(c.num_layers(), 1);
    CHECK(c.has_layer(1));
    CHECK_FALSE(c.has_layer(2));
    CHECK_EQ(c.num_elements(), 1);
    CHECK_EQ(c.num_elements_on_layer(1), 1);
    CHECK_EQ(c.num_elements_on_layer(2), 0);
    CHECK_EQ(c.num_cell_refs(), 0);
    CHECK_EQ(c.num_objects(), 1);
    odrc::db::element e2;
    e2.set_layer(2);
    c.insert(e2);
    CHECK_EQ(c.num_layers(), 2);
    CHECK(c.has_layer(1));
    CHECK(c.has_layer(2));
    CHECK_EQ(c.num_elements(), 2);
    CHECK_EQ(c.num_elements_on_layer(1), 1);
    CHECK_EQ(c.num_elements_on_layer(2), 1);
    CHECK_EQ(c.num_cell_refs(), 0);
    CHECK_EQ(c.num_objects(), 2);
  }
  TEST_CASE("test different element type") {
    using gs     = odrc::geometry::geometry_space<double>;
    using elem_t = odrc::db::element<odrc::geometry::polygon<gs>>;
    odrc::db::cell<elem_t> c;
    CHECK_EQ(c.num_elements(), 0);
    elem_t e;
    c.insert(e);
    CHECK_EQ(c.num_elements(), 1);
  }
}

TEST_SUITE("[OpenDRC] odrc::db cell_ref tests") {
  TEST_CASE("test cell_ref") {
    odrc::geometry::point p(1, 2);
    odrc::db::cell_ref<>  cref(3, p);
    CHECK_EQ(cref.get_id(), 3);
    CHECK_EQ(cref.get_ref_point(), p);
  }
}