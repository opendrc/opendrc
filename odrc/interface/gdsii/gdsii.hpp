#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <type_traits>

namespace odrc::gdsii {
enum class record_type : std::underlying_type_t<std::byte> {
  HEADER = 0x00,
  BGNLIB = 0x01,
  LIBNAME = 0x02,
  UNITS = 0x03,
  ENDLIB = 0x04,
};
enum class data_type : std::underlying_type_t<std::byte> {
  no_data = 0x00,
  int16 = 0x02,
  int32 = 0x03,
  real32 = 0x04,
  real64 = 0x05,
  ascii_string = 0x06,
};

// Data parsers
int16_t parse_int16(const std::byte* bytes);
double parse_real64(const std::byte* bytes);

class library {
 public:
  struct datetime {
    int year = -1;
    int month = -1;
    int day = -1;
    int hour = -1;
    int minute = -1;
    int second = -1;
  };
  void read(const std::filesystem::path& file_path);
  // meta info
  int version = -1;
  datetime mtime;
  datetime atime;
  std::string name;
  double dbu_in_user_unit;
  double dbu_in_meter;
};
}  // namespace odrc::gdsii