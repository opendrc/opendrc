#pragma once

#include <algorithm>
#include <bitset>
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
  AREF     = 0x0B,
  LAYER    = 0x0D,
  DATATYPE = 0x0E,
  WIDTH    = 0x0F,
  XY       = 0x10,
  ENDEL    = 0x11,
  SNAME    = 0x12,
  COLROW   = 0x13,
  NODE     = 0x15,
  STRANS   = 0x1A,
  MAG      = 0x1B,
  ANGLE    = 0x1C,
  PATHTYPE = 0x21,
  NODETYPE = 0x2A,
  BOX      = 0x2D,
  BOXTYPE  = 0x2E,
  BGNEXTN  = 0x30,
  ENDEXTN  = 0x31,
};
enum class data_type : std::underlying_type_t<std::byte> {
  no_data      = 0x00,
  bit_array    = 0x01,
  int16        = 0x02,
  int32        = 0x03,
  real32       = 0x04,
  real64       = 0x05,
  ascii_string = 0x06,
};

// Data parsers
int16_t         parse_int16(const std::byte* bytes);
int32_t         parse_int32(const std::byte* bytes);
double          parse_real64(const std::byte* bytes);
std::bitset<16> parse_bitarray(const std::byte* bytes);
std::string     parse_string(const std::byte* begin, const std::byte* end);

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
    record_type rtype;
    virtual ~element(){};
  };
  struct path : public element {
    int             layer;
    int             datatype;
    int             pathtype;
    int             width;
    int             bgnextn;
    int             endextn;
    std::vector<xy> coordinates;
  };
  struct boundary : public element {
    int             layer;
    int             datatype;
    std::vector<xy> coordinates;
  };
  struct sref : public element {
    bool            angle_flag;
    bool            mag_flag;
    bool            reflection_flag;
    double          mag;
    double          angle;
    std::string     sname;
    std::vector<xy> coordinates;
  };
  struct aref : public element {
    int             columns;
    int             rows;
    bool            angle_flag;
    bool            mag_flag;
    bool            reflection_flag;
    double          mag;
    double          angle;
    std::string     sname;
    std::vector<xy> coordinates;
  };
  struct node : public element {
    int             nodetype;
    int             layer;
    std::vector<xy> coordinates;
  };
  struct box : public element {
    int             boxtype;
    int             layer;
    std::vector<xy> coordinates;
  };
  struct structure {
    datetime              mtime;
    datetime              atime;
    std::string           strname;
    std::vector<element*> elements;
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
  // destructor function
  ~library() {
    for (auto&& n : structs)
      for (auto&& p : n.elements) {
        delete p;
      };
  }

 private:
  datetime _read_time(const std::byte* bytes);
};
}  // namespace odrc::gdsii