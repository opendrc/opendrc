#pragma once

#include <climits>
#include <cstdint>
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
  int                mbr1[4] = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};

  bool is_touching(const polygon& other) const {
    return mbr1[0] < other.mbr1[1] and mbr1[1] > other.mbr1[0] and
           mbr1[2] < other.mbr1[3] and mbr1[3] > other.mbr1[2];
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

struct h_edge {
  int x1;
  int x2;
  int y;
};

struct v_edge {
  int x;
  int y1;
  int y2;
};

class cell_ref {
 public:
  std::string         cell_name;
  coord               ref_point;
  transform           trans;
  int                 mbr1[4] = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
  int                 mbr2[4] = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
  std::vector<h_edge> h_edges;
  std::vector<h_edge> h_edges1;
  std::vector<h_edge> h_edges2;
  std::vector<v_edge> v_edges;
  std::vector<v_edge> v_edges1;
  std::vector<v_edge> v_edges2;

  cell_ref() = default;
  cell_ref(const std::string& name, const coord& p)
      : cell_name(name), ref_point(p) {}

  cell_ref(const std::string& name, const coord& p, const transform& t)
      : cell_name(name), ref_point(p), trans(t) {}

  bool is_touching(const polygon& other) const {
    return mbr1[0] < other.mbr1[1] and mbr1[1] > other.mbr1[0] and
           mbr1[2] < other.mbr1[3] and mbr1[3] > other.mbr1[2];
  }
  bool is_touching(const cell_ref& other) const {
    return mbr1[0] < other.mbr1[1] and mbr1[1] > other.mbr1[0] and
           mbr1[2] < other.mbr1[3] and mbr1[3] > other.mbr1[2];
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
    return mbr1[0] < other.mbr1[1] and mbr1[1] > other.mbr1[0] and
           mbr1[2] < other.mbr1[3] and mbr1[3] > other.mbr1[2];
  }

  bool is_touching(const polygon& other) const {
    return mbr1[0] < other.mbr1[1] and mbr1[1] > other.mbr1[0] and
           mbr1[2] < other.mbr1[3] and mbr1[3] > other.mbr1[2];
  }

  // a bit-wise representation of layers it spans across
  uint64_t              layers = 0;
  std::string           name;
  odrc::util::datetime  mtime;
  odrc::util::datetime  atime;
  std::vector<polygon>  polygons;
  std::vector<cell_ref> cell_refs;
  int                   mbr1[4] = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
  int                   depth   = -1;
};

inline bool polygon::is_touching(const cell& other) const {
  return mbr1[0] < other.mbr1[1] and mbr1[1] > other.mbr1[0] and
         mbr1[2] < other.mbr1[3] and mbr1[3] > other.mbr1[2];
}
}  // namespace odrc::core