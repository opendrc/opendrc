#include <odrc/db/db.hpp>

#include <doctest/doctest.h>

#include <odrc/db/elem.hpp>

TEST_SUITE("[OpenDRC] odrc::db db tests") {
  TEST_CASE("test layer construction") {
    odrc::db::database db;

    auto& cell = db.create_cell("");
    cell.add(odrc::db::element{1, {}});
    cell.add(odrc::db::element{2, {}});
    cell.add(odrc::db::element{2, {}});
    cell.add(odrc::db::element{3, {}});
    db.construct_layers();
    CHECK_EQ(db.layers().size(), 3);
    CHECK_EQ(db.layers().at(1).size(), 1);
    CHECK_EQ(db.layers().at(2).size(), 2);
    CHECK_EQ(db.layers().at(3).size(), 1);
  }
}