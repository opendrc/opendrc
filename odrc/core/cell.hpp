#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <odrc/utility/datetime.hpp>

namespace odrc::core {

struct coord {
  int x;
  int y;
};

class cell;
class polygon {
 public:
  int layer;
  int datatype;

  std::vector<coord> points;
  int                mbr[4] = {};

  bool is_touching(const polygon& other) const {
    return mbr[0] < other.mbr[1] and mbr[1] > other.mbr[0] and
           mbr[2] < other.mbr[3] and mbr[3] > other.mbr[2];
  }
  bool is_touching(const cell& other) const;
};

struct transform {
  bool   is_reflected;
  bool   is_magnified;
  bool   is_rotated;
  double mag;
  double angle;
};

class cell_ref {
 public:
  std::string cell_name;
  coord       ref_point;
  transform   trans;
  int         mbr[4] = {};


  bool is_touching(const polygon& other) const {
    return mbr[0] < other.mbr[1] and mbr[1] > other.mbr[0] and
           mbr[2] < other.mbr[3] and mbr[3] > other.mbr[2];
  }
  bool is_touching(const cell_ref& other) const {
    return mbr[0] < other.mbr[1] and mbr[1] > other.mbr[0] and
           mbr[2] < other.mbr[3] and mbr[3] > other.mbr[2];
  }
};

class cell {
 public:
  polygon&  create_polygon() { return polygons.emplace_back(); }
  cell_ref& create_cell_ref() { return cell_refs.emplace_back(); }

  void add_layer(int layer) {  // might need a better name
    uint64_t mask = 1 << layer;
    add_layer_with_mask(mask);
  }

  void add_layer_with_mask(uint64_t mask) { layers |= mask; }

  bool is_touching(int layer) const {  // might need a better name
    uint64_t mask = 1 << layer;
    return (layers & mask) != 0;
  }

  bool is_touching(const cell& other) const {
    return mbr[0] < other.mbr[1] and mbr[1] > other.mbr[0] and
           mbr[2] < other.mbr[3] and mbr[3] > other.mbr[2];
  }

  bool is_touching(const polygon& other) const {
    return mbr[0] < other.mbr[1] and mbr[1] > other.mbr[0] and
           mbr[2] < other.mbr[3] and mbr[3] > other.mbr[2];
  }

  // a bit-wise representation of layers it spans across
  uint64_t              layers = 0;
  std::string           name;
  odrc::util::datetime  mtime;
  odrc::util::datetime  atime;
  std::vector<polygon>  polygons;
  std::vector<cell_ref> cell_refs;
  int                   mbr[4] = {};
  int                   depth  = -1;
};

inline bool polygon::is_touching(const cell& other) const {
  return mbr[0] < other.mbr[1] and mbr[1] > other.mbr[0] and
         mbr[2] < other.mbr[3] and mbr[3] > other.mbr[2];
}
}  // namespace odrc::core