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
  SNAME    = 0x12,//
  COLROW   = 0x13,    
  TEXTNODE = 0x14,///
  NODE     = 0x15,
  TEXTTYPE = 0x16,
  PRESENTATION = 0x17,
  STRING   = 0x19,
  STRANS   = 0x1A,
  MAG      = 0x1B,
  ANGLE    = 0x1C,
  REFLIBS  = 0x1F,
  FONTS    = 0x20,
  PATHTYPE = 0x21,
  GENERATIONS = 0x22,
  ATTRTABLE = 0x23,
  STYPTAPE  = 0x24,
  STRTYPE   = 0X25,
  ELFLAGS   = 0X26,
  ELKEY     = 0X27,
  NODETYPE  = 0X2A,
  PROPATTR  = 0x2B,
  PROPVALUE = 0x2C,
  BOX       = 0x2D,
  BOXTYPE   = 0x2E,
  PLEX      = 0x2F,
  BGNEXTN   = 0x30,
  ENDEXTN   = 0x31,
  TAPENUM   = 0x32,///
  TAPECODE  = 0x33,///
  STRCLASS  = 0x34,///
  RESERVED  = 0x35,///
  FORMAT    = 0x36,
  MASK      = 0x37,
  ENDMASKS  = 0x38,
  LIBDIRSIZE= 0x39,
  SRFNAME   = 0x3A,
  LIBSECUR  = 0x3B,
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

  struct colrow {
    int columns = -1;
    int rows = -1;
  };

  struct stran {
    int srtrans = -1;
    double mag= -1;
    double angle=-1;
  };

  struct property {
    int propatter = -1;
    std::string propvalue;
  };
  
  struct textbody {
    int texttype = -1;
    int presentation = -1;
    int pathtype = -1;
    int width=-1;
    std::vector<stran> strans;
    std::vector<xy> points;
    std::string strings;
  };

  struct element {
    record_type     rtype;
    int             layer;
    int             datatype;
    int             nodetype;
    int             boxtype;
    int             elflags;
    int             pathtype;
    long            plex;
    long            width;
    long            bgnextn;
    long            endextn;
    std::string     sname;
    std::vector<textbody> textbodys;
    std::vector<colrow> colrows;
    std::vector<xy> points;
    std::vector<property>propertys;
  };

  struct format_type {
    int            format;
    std::string    mask;
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
  int         generations=-1;
  int         libdirsize=-1;
  int         libsecur=-1; 
  datetime    mtime;
  datetime    atime;
  std::string name;
  std::string reflibs;
  std::string fonts;
  std::string attrtable;
  std::string srfname;     
  double      dbu_in_user_unit;
  double      dbu_in_meter;

  // structure definition
  std::vector<format_type> formats;
  std::vector<structure> structs;
  // structure instantiation
  std::vector<std::pair<structure*, xy>> instances;

 private:
  datetime _read_time(const std::byte* bytes);
};
}  // namespace odrc::gdsii

//运行新的代码
//写验证程序