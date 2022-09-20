#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace odrc::gdsii {
enum class record_type : std::underlying_type_t<std::byte> {
  HEADER   = 0x00,
  BGNLIB   = 0x01,
  LIBNAME  = 0x02,
  UNITS    = 0x03,
  ENDLIB   = 0x04,
  BGNSTR   = 0x05,
  STRNAME  = 0x06,
  ENDSTR   = 0x07,
  BOUNDARY = 0x08,
  PATH     = 0x09,
  SREF     = 0x0A,
  LAYER    = 0x0D,
  DATATYPE = 0x0E,
  XY       = 0x10,
  ENDEL    = 0x11,
  SNAME    = 0x12,
};
enum class data_type : std::underlying_type_t<std::byte> {
  no_data      = 0x00,
  int16        = 0x02,
  int32        = 0x03,
  real32       = 0x04,
  real64       = 0x05,
  ascii_string = 0x06,
};

// Data parsers
int16_t     parse_int16(const std::byte* bytes);
int32_t     parse_int32(const std::byte* bytes);
double      parse_real64(const std::byte* bytes);
std::string parse_string(const std::byte* begin, const std::byte* end);

class library {
 public:
  struct datetime {
    int year   = -1;
    int month  = -1;
    int day    = -1;
    int hour   = -1;
    int minute = -1;
    int second = -1;
  };
  struct xy {
    int x = -1;
    int y = -1;
  };
  struct element {
    record_type     rtype;
    int             layer;
    int             datatype;
    std::vector<xy> points;
  };
  struct structure {
    datetime             mtime;
    datetime             atime;
    std::string          name;
    std::vector<element> elements;
  };
  void read(const std::filesystem::path& file_path);
  // meta info
  int         version = -1;
  datetime    mtime;
  datetime    atime;
  std::string name;
  double      dbu_in_user_unit;
  double      dbu_in_meter;

  // structure definition
  std::vector<structure> structs;

  // structure instantiation
  std::vector<std::pair<structure*, xy>> instances;

 private:
  datetime _read_time(const std::byte* bytes);
};
}  // namespace odrc::gdsii