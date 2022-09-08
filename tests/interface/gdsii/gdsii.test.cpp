#include <odrc/interface/gdsii/gdsii.hpp>

#include <cstddef>
#include <exception>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc::gdsii data parser tests") {
  using B = std::byte;
  TEST_CASE("parse_int16 of 0x0001") {
    B bytes[2]{B{0x00}, B{0x01}};
    CHECK_EQ(odrc::gdsii::parse_int16(bytes), int16_t{1});
  }
  TEST_CASE("parse_int16 of 0x0002") {
    B bytes[2]{B{0x00}, B{0x02}};
    CHECK_EQ(odrc::gdsii::parse_int16(bytes), int16_t{2});
  }
  TEST_CASE("parse_int16 of 0x0089") {
    B bytes[2]{B{0x00}, B{0x89}};
    CHECK_EQ(odrc::gdsii::parse_int16(bytes), int16_t{137});
  }
  TEST_CASE("parse_int16 of 0xffff") {
    B bytes[2]{B{0xff}, B{0xff}};
    CHECK_EQ(odrc::gdsii::parse_int16(bytes), int16_t{-1});
  }
  TEST_CASE("parse_int16 of 0xff77") {
    B bytes[2]{B{0xff}, B{0x77}};
    CHECK_EQ(odrc::gdsii::parse_int16(bytes), int16_t{-137});
  }
  TEST_CASE("parse_real64 of 0x41100000") {
    B bytes[8]{B{0x41}, B{0x10}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{1}));
  }
  TEST_CASE("parse_real64 of 0x41200000") {
    B bytes[8]{B{0x41}, B{0x20}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{2}));
  }
  TEST_CASE("parse_real64 of 0x41300000") {
    B bytes[8]{B{0x41}, B{0x30}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{3}));
  }
  TEST_CASE("parse_real64 of 0xc1100000") {
    B bytes[8]{B{0xc1}, B{0x10}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{-1}));
  }
  TEST_CASE("parse_real64 of 0xc1200000") {
    B bytes[8]{B{0xc1}, B{0x20}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{-2}));
  }
  TEST_CASE("parse_real64 of 0xc1300000") {
    B bytes[8]{B{0xc1}, B{0x30}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{-3}));
  }
  TEST_CASE("parse_real64 of 0x40800000") {
    B bytes[8]{B{0x40}, B{0x80}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{0.5}));
  }
  TEST_CASE("parse_real64 of 0x40999999") {
    B bytes[8]{B{0x40}, B{0x99}, B{0x99}, B{0x99}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{0.6}));
  }
  TEST_CASE("parse_real64 of 0x40b33333") {
    B bytes[8]{B{0x40}, B{0xb3}, B{0x33}, B{0x33}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{0.7}));
  }
  TEST_CASE("parse_real64 of 0x41180000") {
    B bytes[8]{B{0x41}, B{0x18}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{1.5}));
  }
  TEST_CASE("parse_real64 of 0x41199999") {
    B bytes[8]{B{0x41}, B{0x19}, B{0x99}, B{0x99}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{1.6}));
  }
  TEST_CASE("parse_real64 of 0x411b3333") {
    B bytes[8]{B{0x41}, B{0x1b}, B{0x33}, B{0x33}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{1.7}));
  }
}

TEST_SUITE("[OpenDRC] odrc::gdsii library tests") {
  TEST_CASE("read normal gdsii file") {
    odrc::gdsii::library lib;
    lib.read("./gcd.gds");
    CHECK_EQ(lib.version, 600);
    CHECK_EQ(lib.dbu_in_meter / lib.dbu_in_user_unit, doctest::Approx(1e-6));
  }
  TEST_CASE("open gdsii file error") {
    odrc::gdsii::library lib;
    CHECK_THROWS_AS(lib.read("./not_exist.gds"), std::runtime_error);
  }
}