#include <odrc/interface/gdsii/gdsii.hpp>

#include <exception>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc/interface/gdsii tests") {
  TEST_CASE("[OpenDRC] read normal gdsii file") {
    odrc::gdsii_library lib;
    lib.read("./gcd.gds");
    CHECK_EQ(lib.gdsii_version, 600);
  }
  TEST_CASE("[OpenDRC] open gdsii file error") {
    odrc::gdsii_library lib;
    CHECK_THROWS_AS(lib.read("./not_exist.gds"), std::runtime_error);
  }
}