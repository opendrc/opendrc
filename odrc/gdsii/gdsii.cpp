#include <odrc/gdsii/gdsii.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <vector>

#include <odrc/utility/exception.hpp>

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

odrc::util::datetime parse_datetime(const std::byte* bytes) {
  odrc::util::datetime dt;
  dt.year   = parse_int16(&bytes[0]);
  dt.month  = parse_int16(&bytes[2]);
  dt.day    = parse_int16(&bytes[4]);
  dt.hour   = parse_int16(&bytes[6]);
  dt.minute = parse_int16(&bytes[8]);
  dt.second = parse_int16(&bytes[10]);
  return dt;
}

odrc::core::coord parse_coord(const std::byte* bytes) {
  return odrc::core::coord{parse_int32(&bytes[0]), parse_int32(&bytes[4])};
}

odrc::core::database read(const std::filesystem::path& file_path) {
  odrc::core::database db;
  std::byte            buffer[65536];
  std::ifstream        ifs(file_path, std::ios::in | std::ios::binary);
  if (not ifs) {
    throw odrc::open_file_error("Cannot open " + file_path.string() + ": " +
                                std::strerror(errno));
  }

  // structure size in bytes for indexing

  constexpr int bytes_per_record_head = 4;
  constexpr int bytes_per_real64      = 8;
  constexpr int bytes_per_coord       = 8;
  constexpr int bytes_per_datetime    = 12;

  // variables to help track current element

  record_type           current_element = record_type::ENDLIB;
  odrc::core::cell*     cell            = nullptr;
  odrc::core::polygon*  polygon         = nullptr;
  odrc::core::cell_ref* cell_ref        = nullptr;

  while (true) {
    // read record header
    ifs.read(reinterpret_cast<char*>(buffer), bytes_per_record_head);

    int         record_length = parse_int16(&buffer[0]);
    record_type rtype         = static_cast<record_type>(buffer[2]);
    data_type   dtype         = static_cast<data_type>(buffer[3]);

    auto begin = &buffer[bytes_per_record_head];
    auto end   = &buffer[record_length];

    ifs.read(reinterpret_cast<char*>(begin),
             record_length - bytes_per_record_head);

    switch (rtype) {
      case record_type::HEADER:
        if (dtype != data_type::int16) {
          throw std::runtime_error("ERROR:Data type of HEADER is not int16");
        }
        db.version = parse_int16(begin);
        break;
      case record_type::BGNLIB:
        if (dtype != data_type::int16) {
          throw std::runtime_error("ERROR:Data type of BGNLIB is not int16");
        }
        db.mtime = parse_datetime(begin);
        db.atime = parse_datetime(begin + bytes_per_datetime);
        break;
      case record_type::LIBNAME:
        if (dtype != data_type::ascii_string) {
          throw std::runtime_error("ERROR:Data type of LIBNAME is not string");
        }
        db.name.assign(parse_string(begin, end));
        break;
      case record_type::UNITS:
        if (dtype != data_type::real64) {
          throw std::runtime_error("ERROR:Data type of UNITS is not real64");
        }
        db.dbu_in_user_unit = parse_real64(begin);
        db.dbu_in_meter     = parse_real64(begin + bytes_per_real64);
        break;
      case record_type::ENDLIB:
        if (dtype != data_type::no_data) {
          throw std::runtime_error("ERROR:Data type of ENDLIB is not no_data");
        }
        break;

        // structure-level records

      case record_type::BGNSTR:
        if (dtype != data_type::int16) {
          throw std::runtime_error("ERROR:Data type of BGNSTR is not int16");
        }
        cell        = &db.create_cell();
        cell->mtime = parse_datetime(begin);
        cell->atime = parse_datetime(begin + bytes_per_datetime);
        break;
      case record_type::STRNAME:
        if (dtype != data_type::ascii_string) {
          throw std::runtime_error("ERROR:Data type of STRNAME is not string");
        }
        cell->name.assign(parse_string(begin, end));
        break;
      case record_type::ENDSTR:
        if (dtype != data_type::no_data) {
          throw std::runtime_error("ERROR:Data type of ENDSTR is not no_data");
        }
        cell = nullptr;  // invalidate pointer for safety
        break;

        // elements

      case record_type::BOUNDARY:
        if (dtype != data_type::no_data) {
          throw std::runtime_error(
              "ERROR:Data type of BOUNDARY is not no_data");
        }
        current_element = rtype;
        polygon         = &cell->create_polygon();
        break;
      case record_type::PATH:
        //assert(dtype == data_type::no_data);
        current_element = rtype;
        break;                
      case record_type::SREF:
        if (dtype != data_type::no_data) {
          throw std::runtime_error("ERROR:Data type of SREF is not no_data");
        }
        current_element = rtype;
        cell_ref        = &cell->create_cell_ref();
        break;
      case record_type::AREF:
        if (dtype == data_type::no_data)
        current_element = rtype;
        break;
      case record_type::LAYER:
        if (dtype != data_type::int16) {
          throw std::runtime_error("ERROR:Data type of LAYER is not int16");
        }
        if (current_element == record_type::BOUNDARY) {
          int layer      = parse_int16(begin);
          polygon->layer = layer;
          cell->add_layer(layer);
        }
        break;
      case record_type::DATATYPE:
        if (dtype != data_type::int16) {
          throw std::runtime_error("ERROR:Data type of DATATYPE is not int16");
        }
        if (current_element == record_type::BOUNDARY) {
          polygon->datatype = parse_int16(begin);
        }
        break;
      case record_type::WIDTH:
        assert(dtype == data_type::int32);
        assert(current_element == record_type::PATH);
        break;      
      case record_type::XY:
        if (dtype != data_type::int32) {
          throw std::runtime_error("ERROR:Data type of XY is not int32");
        }
        if (current_element == record_type::BOUNDARY) {
          int num_coords =
              (record_length - bytes_per_record_head) / bytes_per_coord;
          for (int i = 0; i < num_coords; ++i) {
            polygon->points.emplace_back(
                parse_coord(begin + bytes_per_coord * i));
          }
        } else if (current_element == record_type::SREF) {
          // sref contains exactly 1 coordinate
          if (record_length != bytes_per_coord + bytes_per_record_head) {
            throw std::runtime_error("ERROR:Record length is no true");
          }
          cell_ref->ref_point = parse_coord(begin);
        }
        break;
      case record_type::ENDEL:
        if (dtype != data_type::no_data) {
          throw std::runtime_error("ERROR:Data type of ENDEL is not ENDEL");
        }
        // invalidate pointers for safety
        current_element = rtype;
        polygon         = nullptr;
        cell_ref        = nullptr;
        break;
      case record_type::SNAME:
        if (dtype != data_type::ascii_string) {
          throw std::runtime_error("ERROR:Data type of SNAME is not string");
        }
        if (current_element == record_type::SREF) {
          cell_ref->cell_name = parse_string(begin, end);
        }
        break;
      case record_type::COLROW:
        assert(dtype == data_type::int16);
        assert(current_element == record_type::AREF);
        break;
      case record_type::NODE:
        assert(dtype == data_type::no_data);
        current_element = rtype;
        break;   
      case record_type::STRANS:
        if (dtype != data_type::bit_array) {
          throw std::runtime_error(
              "ERROR:Data type of STRANS is not bit_array");
        }
        if (current_element == record_type::SREF) {
          auto strans                  = parse_bitarray(begin);
          cell_ref->trans.is_reflected = strans.test(15);  // 0-th bit from left
          cell_ref->trans.is_magnified = strans.test(2);  // 13-th bit from left
          cell_ref->trans.is_rotated   = strans.test(1);  // 14-th bit from left
        }
        break;
      case record_type::MAG:
        if (dtype != data_type::real64) {
          throw std::runtime_error("ERROR:Data type of MAG is not real64");
        }
        if (current_element == record_type::SREF) {
          cell_ref->trans.mag = parse_real64(begin);
        }
        break;
      case record_type::ANGLE:
        if (dtype != data_type::real64) {
          throw std::runtime_error("ERROR:Data type of ANGLE is not real64");
        }
        if (current_element == record_type::SREF) {
          cell_ref->trans.angle = parse_real64(begin);
        }
        break;
      case record_type::PATHTYPE:
        //assert(dtype == data_type::int16);
        //assert(current_element == record_type::PATH);
        break;
      case record_type::NODETYPE:
        //assert(dtype == data_type::int16);
        //assert(current_element == record_type::NODE);
        break;
      case record_type::BOX:
        //assert(dtype == data_type::no_data);
        current_element = rtype;
        break;
      case record_type::BOXTYPE:
        //assert(dtype == data_type::int16);
        //assert(current_element == record_type::BOX);
        break;
      case record_type::BGNEXTN:
        //assert(dtype == data_type::int32);
        //assert(current_element == record_type::PATH);
        break;
      case record_type::ENDEXTN:
        //assert(dtype == data_type::int32);
        //assert(current_element == record_type::PATH);
        break;
      
      default:
        break;
    }
    if (rtype == record_type::ENDLIB) {
      break;
    }
  }
  return db;
}

}  // namespace odrc::gdsii

namespace odrc::core {
void database::layer_sort() {
  for (const auto& cel : cells) {
    cell_layer.insert(std::pair(cel.name, 0));
    for (const auto& pol : cel.polygons) {
      cell_layer[cel.name] = cell_layer[cel.name] | 1 << pol.layer;
    }
    for (const auto& ref : cel.cell_refs) {
      cell_layer[cel.name] = cell_layer[cel.name] | cell_layer[ref.cell_name];
    }
  }
}

void database::min_bound() {
  for (const auto& cel : cells) {
    int x_max = 0;
    int x_min = 0;
    int y_max = 0;
    int y_min = 0;
    for (const auto& pol : cel.polygons) {
      x_min = std::min(
          x_min, std::min_element(pol.points.begin(), pol.points.end(),
                                  [](auto a, auto b) { return a.x < b.x; })
                     ->x);
      x_max = std::max(
          x_max, std::max_element(pol.points.begin(), pol.points.end(),
                                  [](auto& a, auto& b) { return a.x > b.x; })
                     ->x);
      y_min = std::min(
          y_min, std::min_element(pol.points.begin(), pol.points.end(),
                                  [](auto a, auto b) { return a.y < b.y; })
                     ->y);
      y_max = std::max(
          y_max, std::max_element(pol.points.begin(), pol.points.end(),
                                  [](auto a, auto b) { return a.y > b.y; })
                     ->y);
    }
    for (const auto& ref : cel.cell_refs) {
      x_min = std::min(x_min, min_bound_rect[ref.cell_name].at(0));
      x_max = std::max(x_max, min_bound_rect[ref.cell_name].at(1));
      y_min = std::min(y_min, min_bound_rect[ref.cell_name].at(2));
      y_max = std::max(y_max, min_bound_rect[ref.cell_name].at(3));
    }
    min_bound_rect.insert(
        std::pair(cel.name, std::vector<int>{x_min, x_max, y_min, y_max}));
  }
}
}  // namespace odrc::core
