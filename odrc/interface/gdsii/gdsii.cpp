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

std::bitset<16> parse_bitarray(const std::byte* bytes) {
  return (std::bitset<16>(parse_int16(bytes)));
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
        structs.back().name.assign(
            parse_string(&buffer[4], &buffer[record_length]));
        break;
      case record_type::ENDSTR:
        assert(dtype == data_type::no_data);
        break;
      case record_type::BOUNDARY:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        structs.back().elements.emplace_back();
        structs.back().elements.back().rtype = rtype;
        break;
      case record_type::PATH:
        assert(dtype == data_type::no_data);
        current_stream = rtype;
        structs.back().elements.emplace_back();
        structs.back().elements.back().rtype = rtype;
        break;
      case record_type::SREF:
        assert(dtype == data_type::no_data);
        structs.back().elements.emplace_back();
        current_stream = rtype;
        break;
      case record_type::AREF:
        assert(dtype == data_type::no_data);
        structs.back().elements.emplace_back();
        current_stream = rtype;
        break;
      case record_type::LAYER:
        assert(dtype == data_type::int16);
        structs.back().elements.back().layer = parse_int16(&buffer[4]);
        break;
      case record_type::DATATYPE:
        assert(dtype == data_type::int16);
        structs.back().elements.back().datatype = parse_int16(&buffer[4]);
        break;
      case record_type::XY:
        assert(dtype == data_type::int32);
        if (current_stream == record_type::SREF) {
          int x = parse_int32(&buffer[4]);
          int y = parse_int32(&buffer[8]);
          structs.back().elements.back().points.emplace_back(xy{x, y});
        } else if (current_stream == record_type::AREF) {
          for (int i = 0; i < 3; ++i) {
            int x = parse_int32(&buffer[4 + i * 8]);
            int y = parse_int32(&buffer[8 + i * 8]);
            structs.back().elements.back().points.emplace_back(xy{x, y});
          }
        } else {
          int num_coors = (record_length - 4) / 8;
          for (int i = 0; i < num_coors; ++i) {
            int x = parse_int32(&buffer[4 + i * 8]);
            int y = parse_int32(&buffer[8 + i * 8]);
            structs.back().elements.back().points.emplace_back(xy{x, y});
          }
        }
        break;
      case record_type::ENDEL:
        assert(dtype == data_type::no_data);
        break;
      case record_type::SNAME: {
        assert(dtype == data_type::ascii_string);
        std::string sname = parse_string(&buffer[4], &buffer[record_length]);
        if (current_stream == record_type::SREF or
            current_stream == record_type::AREF) {
          for (auto&& s : structs) {
            if (s.name == sname) {
              instances.emplace_back(
                  std::make_pair(&s, &(structs.back().elements.back())));
              break;
            }
          }
        }
      } break;
      case record_type::COLROW:
        assert(dtype == data_type::int16);
        {
          structs.back().elements.back().columns = parse_int16(&buffer[4]);
          structs.back().elements.back().rows    = parse_int16(&buffer[6]);
        }
        break;
      case record_type::STRANS:
        assert(dtype == data_type::bit_array);
        if (current_stream == record_type::AREF or
            current_stream == record_type::SREF) {
          structs.back().elements.back().strans = parse_bitarray(&buffer[4]);
        }
        break;
      case record_type::MAG:
        assert(dtype == data_type::real64);
        if (current_stream == record_type::AREF or
            current_stream == record_type::SREF) {
          structs.back().elements.back().mag = parse_real64(&buffer[4]);
        }
        break;
      case record_type::ANGLE:
        assert(dtype == data_type::real64);
        if (current_stream == record_type::AREF or
            current_stream == record_type::SREF) {
          structs.back().elements.back().angle = parse_real64(&buffer[4]);
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