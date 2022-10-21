#include <odrc/interface/gdsii/gdsii.hpp>

#include <doctest/doctest.h>
#include <math.h>
#include <bitset>
#include <cstddef>
#include <exception>

TEST_SUITE("[OpenDRC] odrc::gdsii data parser tests") {
  using B = std::byte;
  TEST_CASE("parse_bitarray of 0x0001") {
    B bytes[2]{B{0x00}, B{0x01}};
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(15), false);
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(2), false);
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(0), true);
  }
  TEST_CASE("parse_bitarray of 0x0005") {
    B bytes[2]{B{0x00}, B{0x05}};
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(15), false);
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(2), true);
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(0), true);
  }
  TEST_CASE("parse_bitarray of 0x8000") {
    B bytes[2]{B{0x80}, B{0x00}};
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(15), true);
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(2), false);
    CHECK_EQ(odrc::gdsii::parse_bitarray(bytes).test(0), false);
  }
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
  TEST_CASE("parse_int32 of 0x00000001") {
    B bytes[4]{B{0x00}, B{0x00}, B{0x00}, B{0x01}};
    CHECK_EQ(odrc::gdsii::parse_int32(bytes), int32_t{1});
  }
  TEST_CASE("parse_int32 of 0x00000002") {
    B bytes[4]{B{0x00}, B{0x00}, B{0x00}, B{0x02}};
    CHECK_EQ(odrc::gdsii::parse_int32(bytes), int32_t{2});
  }
  TEST_CASE("parse_int32 of 0x00000089") {
    B bytes[4]{B{0x00}, B{0x00}, B{0x00}, B{0x89}};
    CHECK_EQ(odrc::gdsii::parse_int32(bytes), int32_t{137});
  }
  TEST_CASE("parse_int32 of 0xffffffff") {
    B bytes[4]{B{0xff}, B{0xff}, B{0xff}, B{0xff}};
    CHECK_EQ(odrc::gdsii::parse_int32(bytes), int32_t{-1});
  }
  TEST_CASE("parse_int32 of 0xfffffffe") {
    B bytes[4]{B{0xff}, B{0xff}, B{0xff}, B{0xfe}};
    CHECK_EQ(odrc::gdsii::parse_int32(bytes), int32_t{-2});
  }
  TEST_CASE("parse_int32 of 0xffffff77") {
    B bytes[4]{B{0xff}, B{0xff}, B{0xff}, B{0x77}};
    CHECK_EQ(odrc::gdsii::parse_int32(bytes), int32_t{-137});
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
  TEST_CASE("parse_string of 'hello world'") {
    B           bytes[11];
    std::string golden_str = "hello world";
    for (int i = 0; i < 11; ++i) {
      bytes[i] = std::byte{static_cast<unsigned char>(golden_str[i])};
    }
    std::string str = odrc::gdsii::parse_string(bytes, bytes + 11);
    CHECK_EQ(str, golden_str);
  }
  TEST_CASE("parse_string of 'hello world' with padded zero") {
    B           bytes[12];
    std::string golden_str = "hello world";
    for (int i = 0; i < 11; ++i) {
      bytes[i] = std::byte{static_cast<unsigned char>(golden_str[i])};
    }
    bytes[11]       = {B{0x00}};
    std::string str = odrc::gdsii::parse_string(bytes, bytes + 12);
    CHECK_EQ(str.length(), 11);
    CHECK_EQ(str, golden_str);
  }
}

TEST_SUITE("[OpenDRC] odrc::gdsii library tests") {
  TEST_CASE("read normal gdsii file") {  // results are from gdstk or klayout
    odrc::gdsii::library lib;
    lib.read("../../data/gcd/gcd.gds");
    CHECK_EQ(lib.version, 600);
    CHECK_EQ(lib.dbu_in_meter / lib.dbu_in_user_unit, doctest::Approx(1e-6));
    CHECK_EQ(lib.structs.size(), 53);
    // CHECK_EQ(lib.instances.size(), 4182);
  }
  TEST_CASE("TEST 00") {
    odrc::gdsii::library lib;
    lib.read("../../data/gcd/test.gds");
    CHECK_EQ(fabs(lib.dbu_in_user_unit / 0.001 - 1.0) < 1e-6, true);
    CHECK_EQ(lib.dbu_in_meter / lib.dbu_in_user_unit, doctest::Approx(1e-6));
    CHECK_EQ(lib.name, "LIB.DB");
    CHECK_EQ(lib.version, 600);
    CHECK_EQ(lib.mtime.year, 2);  // 02-02-08 18:18:28
    CHECK_EQ(lib.mtime.month, 2);
    CHECK_EQ(lib.mtime.day, 8);
    CHECK_EQ(lib.mtime.hour, 18);
    CHECK_EQ(lib.mtime.minute, 18);
    CHECK_EQ(lib.mtime.second, 28);
    CHECK_EQ(lib.atime.year, 2);  // 02-02-08 18:18:28
    CHECK_EQ(lib.atime.month, 2);
    CHECK_EQ(lib.atime.day, 8);
    CHECK_EQ(lib.atime.hour, 18);
    CHECK_EQ(lib.atime.minute, 18);
    CHECK_EQ(lib.atime.second, 28);
    CHECK_EQ(std::string(lib.structs.begin()->strname), "TRANS");
    CHECK_EQ(std::string(lib.structs.at(1).strname), "INV2");
    CHECK_EQ(std::string(lib.structs.at(2).strname), "RINGO");

  }
  TEST_CASE("open gdsii file error") {
    odrc::gdsii::library lib;
    CHECK_THROWS_AS(lib.read("./not_exist.gds"), std::runtime_error);
  }
}