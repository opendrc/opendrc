#include <odrc/interface/gdsii/gdsii.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace odrc::gdsii {

// Data parsers
int16_t parse_int16(const std::byte* bytes) {
  return (std::to_integer<int16_t>(bytes[0]) << 8) |
         std::to_integer<int16_t>(bytes[1]);
}

bool parse_bitarray(const std::byte* bytes, int bit_num) {
  return (parse_int16(bytes) & 1 << (15 - bit_num));
}

int32_t parse_int32(const std::byte* bytes) {
  return (std::to_integer<int32_t>(bytes[0]) << 24) |
         (std::to_integer<int32_t>(bytes[1]) << 16) |
         (std::to_integer<int32_t>(bytes[2]) << 8) |
         std::to_integer<int32_t>(bytes[3]);
}

// SEEE EEEE MMMM MMMM MMMM MMMM MMMM MMMM
double parse_real64(const std::byte* bytes) {
  // intepret bytes as big-endian
  uint64_t data = 0;
  for (int i = 0; i < 8; ++i) {
    data = (data << 8) + std::to_integer<uint64_t>(bytes[i]);
  }

  // 64 is due to exponent excess-format
  // 14 is due to mantissa shift: 2^56 == 16^14
  int    exponent = ((data & 0x7f00'0000'0000'0000) >> 56) - (64 + 14);
  double mantissa = data & 0x00ff'ffff'ffff'ffff;
  double result   = mantissa * std::exp2(exponent * 4);  // m*16^e
  return (data & 0x8000'0000'0000'0000) > 0 ? -result : result;
}

std::string parse_string(const std::byte* begin, const std::byte* end) {
  const char* b = reinterpret_cast<const char*>(begin);
  const char* e = reinterpret_cast<const char*>(end);
  return std::string(b, *(e - 1) == char{0} ? e - 1 : e);
}

void library::read(const std::filesystem::path& file_path) {
  std::vector<std::byte> buffer(65536);
  std::ifstream          ifs(file_path, std::ios::in | std::ios::binary);
  if (not ifs) {
    throw std::runtime_error("Cannot open " + file_path.string() + ": " +
                             std::strerror(errno));
  }
  record_type current_stream;
  while (true) {
    // read record header
    ifs.read(reinterpret_cast<char*>(buffer.data()), 4);
    int         record_length = parse_int16(&buffer[0]);
    record_type rtype         = static_cast<record_type>(buffer[2]);
    data_type   dtype         = static_cast<data_type>(buffer[3]);
    ifs.read(reinterpret_cast<char*>(buffer.data() + 4), record_length - 4);

    switch (rtype) {
      case record_type::HEADER:
        assert(dtype == data_type::int16);
        version = parse_int16(&buffer[4]);
        break;
      case record_type::BGNLIB:
        assert(dtype == data_type::int16);
        mtime = _read_time(&buffer[4]);
        atime = _read_time(&buffer[16]);
        break;
      case record_type::LIBNAME:
        assert(dtype == data_type::ascii_string);
        name.assign(parse_string(&buffer[4], &buffer[record_length]));
        break;
      case record_type::UNITS:
        assert(dtype == data_type::real64);
        dbu_in_user_unit = parse_real64(&buffer[4]);
        dbu_in_meter     = parse_real64(&buffer[12]);
        break;
      case record_type::ENDLIB:
        assert(dtype == data_type::no_data);
        break;
      case record_type::BGNSTR:
        assert(dtype == data_type::int16);
        structs.emplace_back();
        structs.back().mtime = _read_time(&buffer[4]);
        structs.back().atime = _read_time(&buffer[16]);
        break;
      case record_type::STRNAME:
        assert(dtype == data_type::ascii_string);
        structs.back().strname.assign(
            parse_string(&buffer[4], &buffer[record_length]));
        break;
      case record_type::ENDSTR:
        assert(dtype == data_type::no_data);
        break;
      case record_type::BOUNDARY:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        {
          boundary* boundary_ptr_tmp = new boundary;
          element*  element_ptr_tmp  = static_cast<element*>(boundary_ptr_tmp);
          structs.back().elements.emplace_back(element_ptr_tmp);
        }
        break;
      case record_type::PATH:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        {
          path*    path_ptr_tmp    = new path;
          element* element_ptr_tmp = static_cast<element*>(path_ptr_tmp);
          structs.back().elements.emplace_back(element_ptr_tmp);
        }
        break;
      case record_type::SREF:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        {
          sref*    sref_ptr_tmp    = new sref;
          element* element_ptr_tmp = static_cast<element*>(sref_ptr_tmp);
          structs.back().elements.emplace_back(element_ptr_tmp);
        }
        break;
      case record_type::AREF:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        {
          aref*    aref_ptr_tmp    = new aref;
          element* element_ptr_tmp = static_cast<element*>(aref_ptr_tmp);
          structs.back().elements.emplace_back(element_ptr_tmp);
        }
        break;
      case record_type::LAYER:
        assert(dtype == data_type::int16);
        if (current_stream == record_type::BOUNDARY) {
          boundary* ptr_tmp =
              static_cast<boundary*>(structs.back().elements.back());
          ptr_tmp->layer = parse_int16(&buffer[4]);
        } else if (current_stream == record_type::PATH) {
          path* ptr_tmp  = static_cast<path*>(structs.back().elements.back());
          ptr_tmp->layer = parse_int16(&buffer[4]);
        } else if (current_stream == record_type::NODE) {
          node* ptr_tmp  = static_cast<node*>(structs.back().elements.back());
          ptr_tmp->layer = parse_int16(&buffer[4]);
        } else if (current_stream == record_type::BOX) {
          box* ptr_tmp   = static_cast<box*>(structs.back().elements.back());
          ptr_tmp->layer = parse_int16(&buffer[4]);
        }
        break;
      case record_type::DATATYPE:
        assert(dtype == data_type::int16);
        if (current_stream == record_type::BOUNDARY) {
          boundary* ptr_tmp =
              static_cast<boundary*>(structs.back().elements.back());
          ptr_tmp->datatype = parse_int16(&buffer[4]);
        } else if (current_stream == record_type::PATH) {
          path* ptr_tmp = static_cast<path*>(structs.back().elements.back());
          ptr_tmp->datatype = parse_int16(&buffer[4]);
        }
        break;
      case record_type::WIDTH: {
        assert(dtype == data_type::int32);
        path* ptr_tmp  = static_cast<path*>(structs.back().elements.back());
        ptr_tmp->width = parse_int32(&buffer[4]);
      } break;
      case record_type::XY:
        assert(dtype == data_type::int32);
        if (current_stream == record_type::BOUNDARY) {
          boundary* ptr_tmp =
              static_cast<boundary*>(structs.back().elements.back());
          int num_coors = (record_length - 4) / 8;
          for (int i = 0; i < num_coors; ++i) {
            int x = parse_int32(&buffer[4 + i * 8]);
            int y = parse_int32(&buffer[8 + i * 8]);
            (*ptr_tmp).points.emplace_back(xy{x, y});
          }
        } else if (current_stream == record_type::PATH) {
          path* ptr_tmp   = static_cast<path*>(structs.back().elements.back());
          int   num_coors = (record_length - 4) / 8;
          for (int i = 0; i < num_coors; ++i) {
            int x = parse_int32(&buffer[4 + i * 8]);
            int y = parse_int32(&buffer[8 + i * 8]);
            (*ptr_tmp).points.emplace_back(xy{x, y});
          }
        } else if (current_stream == record_type::SREF and
                   record_length == 12) {
          sref* ptr_tmp = static_cast<sref*>(structs.back().elements.back());
          int   x       = parse_int32(&buffer[4]);
          int   y       = parse_int32(&buffer[8]);
          (*ptr_tmp).points.emplace_back(xy{x, y});
        } else if (current_stream == record_type::AREF and
                   record_length == 28) {
          aref* ptr_tmp = static_cast<aref*>(structs.back().elements.back());
          for (int i = 0; i < 3; ++i) {
            int x = parse_int32(&buffer[4 + i * 8]);
            int y = parse_int32(&buffer[8 + i * 8]);
            (*ptr_tmp).points.emplace_back(xy{x, y});
          }
        } else if (current_stream == record_type::NODE) {
          aref* ptr_tmp   = static_cast<aref*>(structs.back().elements.back());
          int   num_coors = (record_length - 4) / 8;
          for (int i = 0; i < num_coors; ++i) {
            int x = parse_int32(&buffer[4 + i * 8]);
            int y = parse_int32(&buffer[8 + i * 8]);
            (*ptr_tmp).points.emplace_back(xy{x, y});
          }
        } else if (current_stream == record_type::BOX and record_length == 44) {
          aref* ptr_tmp = static_cast<aref*>(structs.back().elements.back());
          for (int i = 0; i < 5; ++i) {
            int x = parse_int32(&buffer[4 + i * 8]);
            int y = parse_int32(&buffer[8 + i * 8]);
            (*ptr_tmp).points.emplace_back(xy{x, y});
          }
        }
        break;
      case record_type::ENDEL:
        assert(dtype == data_type::no_data);
        break;
      case record_type::SNAME:
        assert(dtype == data_type::ascii_string);
        if (current_stream == record_type::SREF) {
          sref* ptr_tmp  = static_cast<sref*>(structs.back().elements.back());
          ptr_tmp->sname = parse_string(&buffer[4], &buffer[record_length]);
        } else if (current_stream == record_type::AREF) {
          aref* ptr_tmp  = static_cast<aref*>(structs.back().elements.back());
          ptr_tmp->sname = parse_string(&buffer[4], &buffer[record_length]);
        }
        break;
      case record_type::COLROW:
        assert(dtype == data_type::int16);
        {
          aref* ptr_tmp = static_cast<aref*>(structs.back().elements.back());
          ptr_tmp->columns = parse_int16(&buffer[4]);
          ptr_tmp->rows    = parse_int16(&buffer[6]);
        }
        break;
      case record_type::NODE:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        {
          node*    element_ptr_tmp = new node;
          element* node_ptr_tmp    = static_cast<element*>(element_ptr_tmp);
          structs.back().elements.emplace_back(node_ptr_tmp);
        }
        break;
      case record_type::STRANS:
        assert(dtype == data_type::bit_array);
        if (current_stream == record_type::SREF) {
          sref* ptr_tmp = static_cast<sref*>(structs.back().elements.back());
          ptr_tmp->strans_flag1 = parse_bitarray(&buffer[4], 13);
          ptr_tmp->strans_flag2 = parse_bitarray(&buffer[4], 14);
          ptr_tmp->strans_flag3 = parse_bitarray(&buffer[4], 15);
        } else if (current_stream == record_type::AREF) {
          aref* ptr_tmp = static_cast<aref*>(structs.back().elements.back());
          ptr_tmp->strans_flag1 = parse_bitarray(&buffer[4], 13);
          ptr_tmp->strans_flag2 = parse_bitarray(&buffer[4], 14);
          ptr_tmp->strans_flag3 = parse_bitarray(&buffer[4], 15);
        }
        break;
      case record_type::MAG:
        assert(dtype == data_type::real64);
        if (current_stream == record_type::SREF) {
          sref* ptr_tmp = static_cast<sref*>(structs.back().elements.back());
          ptr_tmp->mag  = parse_real64(&buffer[4]);
        } else if (current_stream == record_type::AREF) {
          aref* ptr_tmp = static_cast<aref*>(structs.back().elements.back());
          ptr_tmp->mag  = parse_real64(&buffer[4]);
        }
        break;
      case record_type::ANGLE:
        assert(dtype == data_type::real64);
        if (current_stream == record_type::SREF) {
          sref* ptr_tmp  = static_cast<sref*>(structs.back().elements.back());
          ptr_tmp->angle = parse_real64(&buffer[4]);
        } else if (current_stream == record_type::AREF) {
          aref* ptr_tmp  = static_cast<aref*>(structs.back().elements.back());
          ptr_tmp->angle = parse_real64(&buffer[4]);
        }
        break;
      case record_type::PATHTYPE:
        assert(dtype == data_type::int16);
        {
          path* ptr_tmp = static_cast<path*>(structs.back().elements.back());
          ptr_tmp->pathtype = parse_int16(&buffer[4]);
        }
        break;
      case record_type::NODETYPE:
        assert(dtype == data_type::int16);
        {
          node* ptr_tmp = static_cast<node*>(structs.back().elements.back());
          ptr_tmp->nodetype = parse_int16(&buffer[4]);
        }
        break;
      case record_type::BOX:
        assert(dtype == data_type::int16);
        current_stream = rtype;
        {
          box*     box_ptr_tmp     = new box;
          element* element_ptr_tmp = static_cast<element*>(box_ptr_tmp);
          structs.back().elements.emplace_back(element_ptr_tmp);
        }
        break;
      case record_type::BOXTYPE:
        assert(dtype == data_type::int16);
        {
          box* ptr_tmp     = static_cast<box*>(structs.back().elements.back());
          ptr_tmp->boxtype = parse_int16(&buffer[4]);
        }
        break;
      case record_type::BGNEXTN:
        assert(dtype == data_type::int16);
        {
          path* ptr_tmp = static_cast<path*>(structs.back().elements.back());
          ptr_tmp->bgnextn = parse_int32(&buffer[4]);
        }
        break;
      case record_type::ENDEXTN:
        assert(dtype == data_type::int16);
        {
          path* ptr_tmp = static_cast<path*>(structs.back().elements.back());
          ptr_tmp->endextn = parse_int32(&buffer[4]);
        }
        break;
      default:
        break;
    }
    if (rtype == record_type::ENDLIB) {
      break;
    }
  }
}
library::datetime library::_read_time(const std::byte* bytes) {
  datetime dt;
  dt.year   = parse_int16(&bytes[0]);
  dt.month  = parse_int16(&bytes[2]);
  dt.day    = parse_int16(&bytes[4]);
  dt.hour   = parse_int16(&bytes[6]);
  dt.minute = parse_int16(&bytes[8]);
  dt.second = parse_int16(&bytes[10]);
  return dt;
}
}  // namespace odrc::gdsii