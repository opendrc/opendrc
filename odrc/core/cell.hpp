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

class polygon {
 public:
  int layer;
  int datatype;

  std::vector<coord> points;
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

  // a bit-wise representation of layers it spans across
  uint64_t              layers = 0;
  std::string           name;
  odrc::util::datetime  mtime;
  odrc::util::datetime  atime;
  std::vector<polygon>  polygons;
  std::vector<cell_ref> cell_refs;
  int                   mbr[4] = {0, 0, 0, 0};
};

}  // namespace odrc::core