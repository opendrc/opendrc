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
  TEST_CASE("parse_real64 of 0x00000000") {
    B bytes[8]{B{0x00}, B{0x00}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{0}));
  }
  TEST_CASE("parse_real64 of 0x4110000") {
    B bytes[8]{B{0x41}, B{0x10}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{1}));
  }
  TEST_CASE("parse_real64 of 0x41a00000") {
    B bytes[8]{B{0x41}, B{0xa0}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{10}));
  }
  TEST_CASE("parse_real64 of 0x42640000") {
    B bytes[8]{B{0x42}, B{0x64}, B{0x00}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{100}));
  }
  TEST_CASE("parse_real64 of 0x433e8000") {
    B bytes[8]{B{0x43}, B{0x3e}, B{0x80}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{1000}));
  }
  TEST_CASE("parse_real64 of 0x44271000") {
    B bytes[8]{B{0x44}, B{0x27}, B{0x10}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{10000}));
  }
  TEST_CASE("parse_real64 of 0x45186a00") {
    B bytes[8]{B{0x45}, B{0x18}, B{0x6a}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{100000}));
  }
  TEST_CASE("parse_real64 of 0x3d68db8bac710cb4") {
    B bytes[8]{B{0x3d}, B{0x68}, B{0xdb}, B{0x8b},
               B{0xac}, B{0x71}, B{0x0c}, B{0xb4}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{0.0001}));
  }
  TEST_CASE("parse_real64 of 0x386df37f675ef6ec") {
    B bytes[8]{B{0x38}, B{0x6d}, B{0xf3}, B{0x7f},
               B{0x67}, B{0x5e}, B{0xf6}, B{0xec}};
    CHECK_EQ(odrc::gdsii::parse_real64(bytes), doctest::Approx(double{1e-10}));
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