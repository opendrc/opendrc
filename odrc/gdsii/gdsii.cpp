#include <odrc/gdsii/gdsii.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <vector>

#include <odrc/utility/exception.hpp>

namespace odrc::gdsii {

const char* record_type_to_string(const record_type rtype) {
  const char* str;
  switch (rtype) {
    case record_type::HEADER:
      str = "HEADER";
      break;
    case record_type::BGNLIB:
      str = "BGNLIB";
      break;
    case record_type::LIBNAME:
      str = "LIBNAME";
      break;
    case record_type::UNITS:
      str = "UNITS";
      break;
    case record_type::ENDLIB:
      str = "ENDLIB";
      break;
    case record_type::BGNSTR:
      str = "BGNSTR";
      break;
    case record_type::STRNAME:
      str = "STRNAME";
      break;
    case record_type::ENDSTR:
      str = "ENDSTR";
      break;
    case record_type::BOUNDARY:
      str = "BOUNDARY";
      break;
    case record_type::PATH:
      str = "PATH";
      break;
    case record_type::SREF:
      str = "SREF";
      break;
    case record_type::AREF:
      str = "AREF";
      break;
    case record_type::LAYER:
      str = "LAYER";
      break;
    case record_type::DATATYPE:
      str = "DATATYPE";
      break;
    case record_type::WIDTH:
      str = "WIDTH";
      break;
    case record_type::XY:
      str = "XY";
      break;
    case record_type::ENDEL:
      str = "ENDEL";
      break;
    case record_type::SNAME:
      str = "SNAME";
      break;
    case record_type::COLROW:
      str = "COLROW";
      break;
    case record_type::NODE:
      str = "NODE";
      break;
    case record_type::STRANS:
      str = "STRANS";
      break;
    case record_type::MAG:
      str = "MAG";
      break;
    case record_type::ANGLE:
      str = "ANGLE";
      break;
    case record_type::PATHTYPE:
      str = "PATHTYPE";
      break;
    case record_type::NODETYPE:
      str = "NODETYPE";
      break;
    case record_type::BOX:
      str = "BOX";
      break;
    case record_type::BOXTYPE:
      str = "BOXTYPE";
      break;
    case record_type::BGNEXTN:
      str = "BGNEXTN";
      break;
    case record_type::ENDEXTN:
      str = "ENDEXTN";
      break;
  }
  return str;
}

const char* data_type_to_string(const data_type dtype) {
  const char* str;
  switch (dtype) {
    case (data_type::no_data):
      str = "no_data";
      break;
    case (data_type::bit_array):
      str = "bit_array";
      break;
    case (data_type::int16):
      str = "int16";
      break;
    case (data_type::int32):
      str = "int32";
      break;
    case (data_type::real32):
      str = "real32";
      break;
    case (data_type::real64):
      str = "real64";
      break;
    case (data_type::ascii_string):
      str = "ascii_string";
      break;
    default:
      break;
  }
  return str;
}

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

  [[maybe_unused]] constexpr int bytes_per_record_head = 4;
  [[maybe_unused]] constexpr int bytes_per_int32       = 4;
  [[maybe_unused]] constexpr int bytes_per_real64      = 8;
  [[maybe_unused]] constexpr int bytes_per_coord       = 8;
  [[maybe_unused]] constexpr int bytes_per_datetime    = 12;

  // variables to help track current element

  record_type           current_element = record_type::ENDLIB;
  odrc::core::cell*     cell            = nullptr;
  odrc::core::polygon*  polygon         = nullptr;
  odrc::core::cell_ref* cell_ref        = nullptr;

  // aliases for datatype

  [[maybe_unused]] constexpr data_type dt_none   = data_type::no_data;
  [[maybe_unused]] constexpr data_type dt_bits   = data_type::bit_array;
  [[maybe_unused]] constexpr data_type dt_int16  = data_type::int16;
  [[maybe_unused]] constexpr data_type dt_int32  = data_type::int32;
  [[maybe_unused]] constexpr data_type dt_real32 = data_type::real32;
  [[maybe_unused]] constexpr data_type dt_real64 = data_type::real64;
  [[maybe_unused]] constexpr data_type dt_string = data_type::ascii_string;

  while (true) {
    // read record header
    ifs.read(reinterpret_cast<char*>(buffer), bytes_per_record_head);

    int         record_length = parse_int16(&buffer[0]);
    record_type rtype         = static_cast<record_type>(buffer[2]);
    data_type   dtype         = static_cast<data_type>(buffer[3]);

    auto begin = &buffer[bytes_per_record_head];
    auto end   = &buffer[record_length];

    // lambda helpers for type checks
    auto check_dtype = [&](data_type expected) {
      if (dtype != expected) {
        auto what_arg = std::string("Invalid file: expected datatype") +
                        data_type_to_string(expected) + " for record type " +
                        record_type_to_string(rtype) + ", got " +
                        data_type_to_string(dtype) + "\n";
        throw odrc::invalid_file(what_arg);
      }
    };

    auto check_element = [&](record_type expected) {
      if (current_element != expected) {
        auto what_arg = std::string("Invalid file: unexpected record type") +
                        record_type_to_string(rtype) + " in element " +
                        record_type_to_string(current_element) +
                        ". (Expected " + record_type_to_string(expected) +
                        ")\n";
        throw odrc::invalid_file(what_arg);
      }
    };

    ifs.read(reinterpret_cast<char*>(begin),
             record_length - bytes_per_record_head);

    switch (rtype) {
      case record_type::HEADER:
        check_dtype(dt_int16);
        db.version = parse_int16(begin);
        break;
      case record_type::BGNLIB:
        check_dtype(dt_int16);
        db.mtime = parse_datetime(begin);
        db.atime = parse_datetime(begin + bytes_per_datetime);
        break;
      case record_type::LIBNAME:
        check_dtype(dt_string);
        db.name.assign(parse_string(begin, end));
        break;
      case record_type::UNITS:
        check_dtype(dt_real64);
        db.dbu_in_user_unit = parse_real64(begin);
        db.dbu_in_meter     = parse_real64(begin + bytes_per_real64);
        break;
      case record_type::ENDLIB:
        check_dtype(dt_none);
        break;

        // structure-level records

      case record_type::BGNSTR:
        check_dtype(dt_int16);
        cell        = &db.create_cell();
        cell->mtime = parse_datetime(begin);
        cell->atime = parse_datetime(begin + bytes_per_datetime);
        break;
      case record_type::STRNAME:
        check_dtype(dt_string);
        cell->name.assign(parse_string(begin, end));
        db.update_map();  // update map when the name is assigned
        break;
      case record_type::ENDSTR:
        check_dtype(dt_none);
        cell = nullptr;  // invalidate pointer for safety
        break;

        // elements

      case record_type::BOUNDARY:
        check_dtype(dt_none);
        current_element = rtype;
        polygon         = &cell->create_polygon();
        break;
      case record_type::PATH:
        check_dtype(dt_none);
        current_element = rtype;
        break;
      case record_type::SREF:
        check_dtype(dt_none);
        current_element = rtype;
        cell_ref        = &cell->create_cell_ref();
        break;
      case record_type::AREF:
        check_dtype(dt_none);
        current_element = rtype;
        break;
      case record_type::LAYER:
        check_dtype(dt_int16);
        if (current_element == record_type::BOUNDARY) {
          int layer      = parse_int16(begin);
          polygon->layer = layer;
          cell->add_layer(layer);
        }
        break;
      case record_type::DATATYPE:
        check_dtype(dt_int16);
        if (current_element == record_type::BOUNDARY) {
          polygon->datatype = parse_int16(begin);
        }
        break;
      case record_type::WIDTH:
        check_dtype(dt_int32);
        check_element(record_type::PATH);
        break;
      case record_type::XY:
        check_dtype(dt_int32);
        if (current_element == record_type::BOUNDARY) {
          int num_coords =
              (record_length - bytes_per_record_head) / bytes_per_coord;
          for (int i = 0; i < num_coords; ++i) {
            auto coord = parse_coord(begin + bytes_per_coord * i);
            polygon->points.emplace_back(coord);
            polygon->mbr1[0] = std::min(polygon->mbr1[0], coord.x);
            polygon->mbr1[1] = std::max(polygon->mbr1[1], coord.x);
            polygon->mbr1[2] = std::min(polygon->mbr1[2], coord.y);
            polygon->mbr1[3] = std::max(polygon->mbr1[3], coord.y);
            cell->mbr1[0]    = std::min(cell->mbr1[0], coord.x);
            cell->mbr1[1]    = std::max(cell->mbr1[1], coord.x);
            cell->mbr1[2]    = std::min(cell->mbr1[2], coord.y);
            cell->mbr1[3]    = std::max(cell->mbr1[3], coord.y);
          }
        } else if (current_element == record_type::SREF) {
          // sref contains exactly 1 coordinate
          if (record_length != bytes_per_coord + bytes_per_record_head) {
            throw odrc::invalid_file(
                "Expected record length of " +
                std::to_string(bytes_per_coord + bytes_per_record_head) +
                " for record XY inside an SREF element, got " +
                std::to_string(record_length) + "\n");
          }
          auto coord           = parse_coord(begin);
          cell_ref->ref_point  = coord;
          const auto& the_cell = db.get_cell(cell_ref->cell_name);
          cell->mbr1[0] = std::min(cell->mbr1[0], coord.x + the_cell.mbr1[0]);
          cell->mbr1[1] = std::max(cell->mbr1[1], coord.x + the_cell.mbr1[1]);
          cell->mbr1[2] = std::min(cell->mbr1[2], coord.y + the_cell.mbr1[2]);
          cell->mbr1[3] = std::max(cell->mbr1[3], coord.y + the_cell.mbr1[3]);
        }
        break;
      case record_type::ENDEL:
        check_dtype(dt_none);
        // invalidate pointers for safety
        current_element = rtype;
        polygon         = nullptr;
        cell_ref        = nullptr;
        break;
      case record_type::SNAME:
        check_dtype(dt_string);
        if (current_element == record_type::SREF) {
          auto name           = parse_string(begin, end);
          cell_ref->cell_name = name;
          cell->add_layer_with_mask(db.get_cell(name).layers);
        }
        break;
      case record_type::COLROW:
        check_dtype(dt_int16);
        check_element(record_type::AREF);
        break;
      case record_type::NODE:
        check_dtype(dt_none);
        current_element = rtype;
        break;
      case record_type::STRANS:
        check_dtype(dt_bits);
        if (current_element == record_type::SREF) {
          auto strans                  = parse_bitarray(begin);
          cell_ref->trans.is_reflected = strans.test(15);  // 0-th bit from left
          cell_ref->trans.is_magnified = strans.test(2);  // 13-th bit from left
          cell_ref->trans.is_rotated   = strans.test(1);  // 14-th bit from left
        }
        break;
      case record_type::MAG:
        check_dtype(dt_real64);
        if (current_element == record_type::SREF) {
          cell_ref->trans.mag = parse_real64(begin);
        }
        break;
      case record_type::ANGLE:
        check_dtype(dt_real64);
        if (current_element == record_type::SREF) {
          cell_ref->trans.angle = parse_real64(begin);
        }
        break;
      case record_type::PATHTYPE:
        check_dtype(dt_int16);
        check_element(record_type::PATH);
        break;
      case record_type::NODETYPE:
        check_dtype(dt_int16);
        check_element(record_type::NODE);
        break;
      case record_type::BOX:
        check_dtype(dt_none);
        current_element = rtype;
        break;
      case record_type::BOXTYPE:
        check_dtype(dt_int16);
        check_element(record_type::BOX);
        break;
      case record_type::BGNEXTN:
        check_dtype(dt_int32);
        check_element(record_type::PATH);
        break;
      case record_type::ENDEXTN:
        check_dtype(dt_int32);
        check_element(record_type::PATH);
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
