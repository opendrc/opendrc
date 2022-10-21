#include <odrc/utility/exception.hpp>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc::exception tests") {
  TEST_CASE("test throw not_implemented_error") {
    CHECK_THROWS_AS(throw odrc::not_implemented_error("test"),
                    odrc::not_implemented_error);
  }

  TEST_CASE("test INVAILD_FILE_ERROR") {
    CHECK_THROWS_AS(throw odrc::invalid_file("test"), odrc::invalid_file);
  }
}
