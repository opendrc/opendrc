#include <odrc/infrastructure/exception/exception.hpp>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc::exception tests") {
  TEST_CASE("test ERROR_NOT_IMPLEMENTED") {
    CHECK_THROWS_AS(throw odrc::utility::not_implemented_exception("test"),
                    odrc::utility::not_implemented_exception);
  }

  TEST_CASE("test ERROR_INVAILD_FILE") {
    CHECK_THROWS_AS(throw odrc::utility::invaild_file_exception("test"),
                    odrc::utility::invaild_file_exception);
  }
}