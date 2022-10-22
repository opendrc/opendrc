#include <odrc/utility/exception.hpp>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc::exception tests") {
  TEST_CASE("test throw not_implemented_error") {
    CHECK_THROWS_WITH_AS(throw odrc::not_implemented_error(),
                         "Function not implemented.",
                         odrc::not_implemented_error);
    CHECK_THROWS_WITH_AS(
        throw odrc::not_implemented_error("test not implemented"),
        "test not implemented", odrc::not_implemented_error);
  }

  TEST_CASE("test throw open_file_error") {
    CHECK_THROWS_WITH_AS(throw odrc::open_file_error(), "Open file error.",
                         odrc::open_file_error);
    CHECK_THROWS_WITH_AS(throw odrc::open_file_error("test open file error"),
                         "test open file error", odrc::open_file_error);
  }

  TEST_CASE("test throw invalid_file") {
    CHECK_THROWS_WITH_AS(throw odrc::invalid_file(), "Invalid file.",
                         odrc::invalid_file);
    CHECK_THROWS_WITH_AS(throw odrc::invalid_file("test invalid file"),
                         "test invalid file", odrc::invalid_file);
  }
}
