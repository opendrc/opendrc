#pragma once

#include <climits>

namespace odrc::core {
struct mbr {
  int x_min = INT_MAX;
  int x_max = INT_MIN;
  int y_min = INT_MAX;
  int y_max = INT_MIN;
};

template <typename T, typename V>
struct interval {
  T l;
  T r;
  V v;

  T    mid() const { return (l + r) / 2; }
  bool contains(const T& p) const { return l <= p and p <= r; }
};

}  // namespace odrc::core

namespace odrc {
using Intvl = core::interval<int, int>;
struct event {
  Intvl intvl;
  int   y;
  bool  is_polygon;
  bool  is_inevent;
};

//
struct check_result {
  int  e11x;  // first_edge_start_x
  int  e11y;  // first_edge_start_y
  int  e12x;  // first_edge_end_x
  int  e12y;  // first_edge_end_y
  int  e21x;  // second_edge_start_x
  int  e21y;  // second_edge_start_y
  int  e22x;  // second_edge_end_x
  int  e22y;  // second_edge_end_y
  bool is_violation = false;
};
enum class rule_type {
  area,
  enclosure,
  extension,
  geometry,
  length,
  lup,
  overlap,
  recommend,       // may be unused
  spacing_both,    // default,both horizontal and vertical edges
  spacing_h_edge,  // only horizontal edges
  spacing_v_edge,  // only vertical edges
  spacing_corner,  // corner-corner
  spacing_center,  // center-center
  spacing_tip,     // tip-tip
  spacing_lup,     // latch-upobject
  sram,
  width,
  aux_not_bend,
  aux_is_rectilinear,
  aux_ensure
};
}  // namespace odrc