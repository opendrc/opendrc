#pragma once

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <filesystem>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <map>

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
std::bitset<16> parse_bitarray(const std::byte* bytes);
int16_t         parse_int16(const std::byte* bytes);
int32_t         parse_int32(const std::byte* bytes);
double          parse_real64(const std::byte* bytes);
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

  // basic elements

  struct element {
    record_type rtype;
  };

  struct path : public element {
    int             layer;
    int             datatype;
    int             pathtype;
    int             width;
    int             bgnextn;
    int             endextn;
    std::vector<xy> points;  // size in [2, 200]
  };
  struct boundary : public element {
    int             layer;
    int             datatype;
    std::vector<xy> points;  // size in [4, 200]
  };
  struct node : public element {
    int             nodetype;  // 0-63
    int             layer;
    std::vector<xy> points;
  };
  struct box : public element {
    int             boxtype;  // 0-63
    int             layer;
    std::vector<xy> points;
  };

  // structure reference

  struct strans {
    bool   is_reflected;
    bool   is_magnified;
    bool   is_rotated;
    double mag;
    double angle;
  };

  struct sref : public element {
    std::string sname;
    xy          ref_point;
    strans      trans;
  };
  struct aref : public element {
    std::string sname;
    int         num_columns;
    int         num_rows;
    xy          bottomleft;
    xy          bottomright;
    xy          topleft;
    strans      trans;
  };

  // structure definition

  struct structure {
    datetime    mtime;
    datetime    atime;
    std::string strname;
    // a list of <record_type, offset-in-internal-storage> pairs
    std::vector<std::pair<record_type, int>> elements;
  };

  std::vector<structure> structs;

  // public API functions

  void read(const std::filesystem::path& file_path);
  void layer_sort(const std::vector<structure>& structs);
  // meta info
  int         version = -1;
  datetime    mtime;
  datetime    atime;
  std::string name;
  double      dbu_in_user_unit;
  double      dbu_in_meter;

 private:
  datetime _read_time(const std::byte* bytes);
  xy       _read_xy(const std::byte* bytes);

  // internal storage for elements
  std::vector<path>     _paths;
  std::vector<boundary> _boundaries;
  std::vector<node>     _nodes;
  std::vector<box>      _boxes;
  std::vector<sref>     _srefs;
  std::vector<aref>     _arefs;
};
/*
class lib_bylayer{
  public:
  multimap<int,void*> _map;
  map _search(const odrc::gdsii::library );
}*/
}  // namespace odrc::gdsii