#include <odrc/db/cell.hpp>

#include <doctest/doctest.h>

#include <odrc/db/elem.hpp>

TEST_SUITE("[OpenDRC] odrc::db cell tests") {
  TEST_CASE("test empty cell") {
    odrc::db::cell c;
    CHECK_EQ(c.num_elements(), 0);
  }
  TEST_CASE("test add element") {
    odrc::db::cell    c;
    odrc::db::element e1;
    c.add(e1);
    CHECK_EQ(c.num_elements(), 1);
    odrc::db::element e2;
    c.add(e2);
    CHECK_EQ(c.num_elements(), 2);
  }
}

TEST_SUITE("[OpenDRC] odrc::db cell_ref tests") {
  TEST_CASE("test cell_ref") {
    auto               c = std::make_shared<odrc::db::cell<>>();
    odrc::db::cell_ref cref(c);
    CHECK_EQ(cref.get(), c.get());
    CHECK_EQ(cref->num_elements(), 0);
    CHECK_EQ((*cref).num_elements(), 0);
  }
}