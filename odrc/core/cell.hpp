#pragma once

#include <climits>
#include <cstdint>
#include <odrc/core/structs.hpp>
#include <odrc/utility/datetime.hpp>
#include <string>
#include <vector>
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
  // TODO mbr
  int mbr[4] = {
      std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
      std::numeric_limits<int>::max(), std::numeric_limits<int>::min()};

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

// TODO
struct h_edge {
  int x1;
  int x2;
  int y;
};

struct v_edge {
  int y1;
  int y2;
  int x;
};

class cell_ref {
 public:
  std::string cell_name;
  coord       ref_point;
  transform   trans;
  cell_mbr    cell_ref_mbr{
      std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
      std::numeric_limits<int>::max(), std::numeric_limits<int>::min()};
  cell_ref() = default;
  cell_ref(const std::string& name, const coord& p)
      : cell_name(name), ref_point(p) {}

  cell_ref(const std::string& name, const coord& p, const transform& t)
      : cell_name(name), ref_point(p), trans(t) {}
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
  bool is_touching(std::vector<int> layers) const {  // might need a better name
    uint64_t mask = 0;
    for (auto layer : layers) {
      mask |= 1 << layer;
    }
    return (mask & this->layers) != 0;
  }

  bool is_touching(const cell& other) const {
    return mbr.x_min < other.mbr.x_max and mbr.x_max > other.mbr.x_min and
           mbr.y_min < other.mbr.y_max and mbr.y_max > other.mbr.y_min;
  }

  bool is_touching(const polygon& other) const {
    return mbr.x_min < other.mbr[1] and mbr.x_max > other.mbr[0] and
           mbr.y_min < other.mbr[3] and mbr.y_max > other.mbr[2];
  }

  // a bit-wise representation of layers it spans across
  uint64_t              layers = 0;
  std::string           name;
  odrc::util::datetime  mtime;
  odrc::util::datetime  atime;
  std::vector<polygon>  polygons;
  std::vector<cell_ref> cell_refs;
  cell_mbr mbr{std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
               std::numeric_limits<int>::max(),
               std::numeric_limits<int>::min()};
};

inline bool polygon::is_touching(const cell& other) const {
  return mbr[0] < other.mbr.x_max and mbr[1] > other.mbr.x_min and
         mbr[2] < other.mbr.y_max and mbr[3] > other.mbr.y_min;
}
class edges {
 public:
  std::vector<std::vector<h_edge>> h_edges;
  std::vector<std::vector<v_edge>> v_edges;

 private:
};
}  // namespace odrc::core