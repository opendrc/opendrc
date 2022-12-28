#pragma once

#include <climits>
#include <cstdint>
#include <odrc/utility/datetime.hpp>
#include <set>
#include <string>
#include <vector>
#include <map>
namespace odrc::core {
struct cell_mbr {
  int x_min = std::numeric_limits<int>::max();
  int x_max = std::numeric_limits<int>::min();
  int y_min = std::numeric_limits<int>::max();
  int y_max = std::numeric_limits<int>::min();
};
struct edge {
  int x_startpoint;
  int y_startpoint;
  int x_endpoint;
  int y_endpoint;
};
struct orthogonal_edge {
  int start_point;
  int end_point;
  int distance;
};

struct coord {
  int x;
  int y;
};

class cell;
class polygon {
 public:
  int                layer;
  int                datatype;
  std::vector<coord> points;
  cell_mbr mbr{std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
               std::numeric_limits<int>::max(),
               std::numeric_limits<int>::min()};

  bool is_touching(const polygon& other) const {
    return mbr.x_min < other.mbr.x_max and mbr.x_max > other.mbr.x_min and
           mbr.y_min < other.mbr.y_max and mbr.y_max > other.mbr.y_min;
  }
  bool is_touching(const cell& other) const;
  void update_mbr() {
    mbr.x_min = std::min(mbr.x_min, points.back().x);
    mbr.x_max = std::max(mbr.x_max, points.back().x);
    mbr.y_min = std::min(mbr.y_min, points.back().y);
    mbr.y_max = std::max(mbr.y_max, points.back().y);
  }
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
  std::string         cell_name;
  coord               ref_point;
  transform           trans;
  std::map<int,std::vector<orthogonal_edge>> h_edges;
  std::map<int,std::vector<orthogonal_edge>> v_edges;

  cell_mbr            cell_ref_mbr{
      std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
      std::numeric_limits<int>::max(), std::numeric_limits<int>::min()};
  cell_ref() = default;
  cell_ref(const std::string& name, const coord& p)
      : cell_name(name), ref_point(p) {}

  cell_ref(const std::string& name, const coord& p, const transform& t)
      : cell_name(name), ref_point(p), trans(t) {}
  void update_mbr(const cell_mbr& mbr) {
    cell_ref_mbr =
        core::cell_mbr{mbr.x_min + ref_point.x, mbr.x_max + ref_point.x,
                       mbr.y_min + ref_point.y, mbr.y_max + ref_point.y};
  }
};

class cell {
 public:
  // a bit-wise representation of layers it spans across
  uint64_t              layers = 0;
  std::string           name;
  odrc::util::datetime  mtime;
  odrc::util::datetime  atime;
  std::vector<polygon>  polygons;
  std::vector<cell_ref> cell_refs;
  std::set<int>         ref_ids;
  cell_mbr mbr{std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
               std::numeric_limits<int>::max(),
               std::numeric_limits<int>::min()};
  polygon& create_polygon() { return polygons.emplace_back(); }
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
    return ( this->layers&mask ) != 0;
  }

  bool is_touching(const cell& other) const {
    return mbr.x_min < other.mbr.x_max and mbr.x_max > other.mbr.x_min and
           mbr.y_min < other.mbr.y_max and mbr.y_max > other.mbr.y_min;
  }

  bool is_touching(const polygon& other) const {
    return mbr.x_min < other.mbr.x_max and mbr.x_max > other.mbr.x_min and
           mbr.y_min < other.mbr.y_max and mbr.y_max > other.mbr.y_min;
  }
  void update_mbr(const cell_mbr& polygon_mbr) {
    mbr.x_min = std::min(mbr.x_min, polygon_mbr.x_min);
    mbr.x_max = std::max(mbr.x_max, polygon_mbr.x_max);
    mbr.y_min = std::min(mbr.y_min, polygon_mbr.y_min);
    mbr.y_max = std::max(mbr.y_max, polygon_mbr.x_max);
  }
  void update_mbr(const cell_mbr& polygon_mbr, const coord& offset) {
    mbr.x_min = std::min(mbr.x_min, polygon_mbr.x_min + offset.x);
    mbr.x_max = std::max(mbr.x_max, polygon_mbr.x_max + offset.x);
    mbr.y_min = std::min(mbr.y_min, polygon_mbr.y_min + offset.y);
    mbr.y_max = std::max(mbr.y_max, polygon_mbr.x_max + offset.y);
  }
};

inline bool polygon::is_touching(const cell& other) const {
  return mbr.x_min < other.mbr.x_max and mbr.x_max > other.mbr.x_min and
         mbr.y_min < other.mbr.y_max and mbr.y_max > other.mbr.y_min;
}

}  // namespace odrc::core