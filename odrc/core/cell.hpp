#pragma once

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

  std::string           name;
  odrc::util::datetime  mtime;
  odrc::util::datetime  atime;
  std::vector<polygon>  polygons;
  std::vector<cell_ref> cell_refs;
};
}  // namespace odrc::core