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
std::bitset<16> parse_bitarray(const std::byte* bytes) {
  return std::bitset<16>(parse_int16(bytes));
}

int16_t parse_int16(const std::byte* bytes) {
  return (std::to_integer<int16_t>(bytes[0]) << 8) |
         std::to_integer<int16_t>(bytes[1]);
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

  record_type current_stream;  // used to track the stream syntax

  // the stream reader, best described as a finite-state machine (FSM)
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

        // structure-level records

      case record_type::BGNSTR: {
        assert(dtype == data_type::int16);
        auto struc  = structs.emplace_back();
        struc.mtime = _read_time(&buffer[4]);
        struc.atime = _read_time(&buffer[16]);
      } break;
      case record_type::STRNAME:
        assert(dtype == data_type::ascii_string);
        structs.back().strname.assign(
            parse_string(&buffer[4], &buffer[record_length]));
        break;
      case record_type::ENDSTR:
        assert(dtype == data_type::no_data);
        break;

        // elements

      case record_type::BOUNDARY:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        _boundaries.emplace_back();
        structs.back().elements.emplace_back(rtype, _boundaries.size() - 1);
        break;
      case record_type::PATH:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        _paths.emplace_back();
        structs.back().elements.emplace_back(rtype, _paths.size() - 1);
        break;
      case record_type::SREF:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        _srefs.emplace_back();
        structs.back().elements.emplace_back(rtype, _srefs.size() - 1);
        break;
      case record_type::AREF:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        _arefs.emplace_back();
        structs.back().elements.emplace_back(rtype, _arefs.size() - 1);
        break;
      case record_type::LAYER: {
        assert(dtype == data_type::int16);
        int layer = parse_int16(&buffer[4]);
        if (current_stream == record_type::BOUNDARY) {
          _boundaries.back().layer = layer;
        } else if (current_stream == record_type::PATH) {
          _paths.back().layer = layer;
        } else if (current_stream == record_type::NODE) {
          _nodes.back().layer = layer;
        } else if (current_stream == record_type::BOX) {
          _boxes.back().layer = layer;
        }
      } break;
      case record_type::DATATYPE: {
        assert(dtype == data_type::int16);
        int datatype = parse_int16(&buffer[4]);
        if (current_stream == record_type::BOUNDARY) {
          _boundaries.back().datatype = datatype;
        } else if (current_stream == record_type::PATH) {
          _paths.back().datatype = datatype;
        }
      } break;
      case record_type::WIDTH: {
        assert(dtype == data_type::int32);
        assert(current_stream == record_type::PATH);
        _paths.back().width = parse_int32(&buffer[4]);
      } break;
      case record_type::XY:
        assert(dtype == data_type::int32);
        if (current_stream == record_type::BOUNDARY) {
          int num_coords = (record_length - 4) / 8;
          for (int i = 0; i < num_coords; ++i) {
            _boundaries.back().points.emplace_back(
                _read_xy(&buffer[4 + i * 8]));
          }
        } else if (current_stream == record_type::PATH) {
          int num_coords = (record_length - 4) / 8;
          for (int i = 0; i < num_coords; ++i) {
            _paths.back().points.emplace_back(_read_xy(&buffer[4 + i * 8]));
          }
        } else if (current_stream == record_type::SREF) {
          assert(record_length == 12);  // sref contains exactly 1 coordinate
          _srefs.back().ref_point = _read_xy(&buffer[4]);
        } else if (current_stream == record_type::AREF) {
          assert(record_length == 28);  // aref contains exactly 3 coordinates
          auto& aref       = _arefs.back();
          aref.bottomleft  = _read_xy(&buffer[4]);
          aref.bottomright = _read_xy(&buffer[12]);
          aref.topleft     = _read_xy(&buffer[20]);
        } else if (current_stream == record_type::NODE) {
          int num_coords = (record_length - 4) / 8;
          for (int i = 0; i < num_coords; ++i) {
            _nodes.back().points.emplace_back(_read_xy(&buffer[4 + i * 8]));
          }
        } else if (current_stream == record_type::BOX) {
          assert(record_length == 44);  // box contains exactly 5 coordinates
          for (int i = 0; i < 5; ++i) {
            _boxes.back().points.emplace_back(_read_xy(&buffer[4 + i * 8]));
          }
        }
        break;
      case record_type::ENDEL:
        assert(dtype == data_type::no_data);
        break;
      case record_type::SNAME: {
        assert(dtype == data_type::ascii_string);
        std::string sname = parse_string(&buffer[4], &buffer[record_length]);
        if (current_stream == record_type::SREF) {
          _srefs.back().sname = sname;
        } else if (current_stream == record_type::AREF) {
          _arefs.back().sname = sname;
        }
      } break;
      case record_type::COLROW:
        assert(dtype == data_type::int16);
        assert(current_stream == record_type::AREF);
        _arefs.back().num_columns = parse_int16(&buffer[4]);
        _arefs.back().num_rows    = parse_int16(&buffer[6]);
        break;
      case record_type::NODE:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        _nodes.emplace_back();
        structs.back().elements.emplace_back(rtype, _nodes.size() - 1);
        break;
      case record_type::STRANS: {
        assert(dtype == data_type::bit_array);
        auto strans       = parse_bitarray(&buffer[4]);
        bool is_reflected = strans.test(15);  // 0-th bit from left
        bool is_magnified = strans.test(2);   // 13-th bit from left
        bool is_rotated   = strans.test(1);   // 14-th bit from left
        if (current_stream == record_type::SREF) {
          _srefs.back().trans.is_reflected = is_reflected;
          _srefs.back().trans.is_magnified = is_magnified;
          _srefs.back().trans.is_rotated   = is_rotated;
        } else if (current_stream == record_type::AREF) {
          _arefs.back().trans.is_reflected = is_reflected;
          _arefs.back().trans.is_magnified = is_magnified;
          _arefs.back().trans.is_rotated   = is_rotated;
        }
      } break;
      case record_type::MAG: {
        assert(dtype == data_type::real64);
        double mag = parse_real64(&buffer[4]);
        if (current_stream == record_type::SREF) {
          _srefs.back().trans.mag = mag;
        } else if (current_stream == record_type::AREF) {
          _arefs.back().trans.mag = mag;
        }
      } break;
      case record_type::ANGLE: {
        assert(dtype == data_type::real64);
        double angle = parse_real64(&buffer[4]);
        if (current_stream == record_type::SREF) {
          _srefs.back().trans.angle = angle;
        } else if (current_stream == record_type::AREF) {
          _arefs.back().trans.angle = angle;
        }
      } break;
      case record_type::PATHTYPE:
        assert(dtype == data_type::int16);
        assert(current_stream == record_type::PATH);
        _paths.back().pathtype = parse_int16(&buffer[4]);
        break;
      case record_type::NODETYPE:
        assert(dtype == data_type::int16);
        assert(current_stream == record_type::NODE);
        _nodes.back().nodetype = parse_int16(&buffer[4]);
        break;
      case record_type::BOX:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        _boxes.emplace_back();
        structs.back().elements.emplace_back(rtype, _boxes.size() - 1);
        break;
      case record_type::BOXTYPE:
        assert(dtype == data_type::int16);
        _boxes.back().boxtype = parse_int16(&buffer[4]);
        break;
      case record_type::BGNEXTN:
        assert(dtype == data_type::int32);
        assert(current_stream == record_type::PATH);
        _paths.back().bgnextn = parse_int32(&buffer[4]);
        break;
      case record_type::ENDEXTN:
        assert(dtype == data_type::int32);
        assert(current_stream == record_type::PATH);
        _paths.back().endextn = parse_int32(&buffer[4]);
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

library::xy library::_read_xy(const std::byte* bytes) {
  return xy{parse_int32(&bytes[0]), parse_int32(&bytes[4])};
}
}  // namespace odrc::gdsii