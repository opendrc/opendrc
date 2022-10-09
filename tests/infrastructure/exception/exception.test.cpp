#include <odrc/infrastructure/exception/exception.hpp>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc::exception tests") {
  TEST_CASE("test NOT_IMPLEMENTED_ERROR") {
    CHECK_THROWS_AS(throw odrc::not_implemented_exception("test"),
                    odrc::not_implemented_exception);
  }

  TEST_CASE("test INVAILD_FILE_ERROR") {
    CHECK_THROWS_AS(throw odrc::invalid_file_exception("test"),
                    odrc::invalid_file_exception);
  }
}
