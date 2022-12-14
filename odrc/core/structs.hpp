#pragma once

#include <limits>

namespace odrc::core {
struct cell_mbr {
  int x_min = std::numeric_limits<int>::max();
  int x_max = std::numeric_limits<int>::min();
  int y_min = std::numeric_limits<int>::max();
  int y_max = std::numeric_limits<int>::min();
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
  enclosure          = 0,
  spacing_both       = 1,  // default,both horizontal and vertical edges
  spacing_h_edge     = 2,  // only horizontal edges
  spacing_v_edge     = 3,  // only vertical edges
  spacing_corner     = 4,  // corner-corner
  spacing_center     = 5,  // center-center
  spacing_tip        = 6,  // tip-tip
  spacing_lup        = 7,  // latch-upobject
  extension          = 8,
  geometry           = 9,
  length             = 10,
  lup                = 11,
  overlap            = 12,
  recommend          = 13,  // may be unused
  area               = 14,
  sram               = 15,
  width              = 16,
  aux_not_bend       = 17,
  aux_is_rectilinear = 18,
  aux_ensure         = 19
};
}  // namespace odrc