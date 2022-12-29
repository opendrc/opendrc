#pragma once

#include <limits>
#include <odrc/core/database.hpp>
#include <odrc/core/interval_tree.hpp>
#include "odrc/core/cell.hpp"
namespace odrc {
using Intvl = core::interval<int, int>;

struct event {
  Intvl intvl;
  int   y;
  bool  is_polygon;
  bool  is_inevent;
};

struct violation_information {
  const core::polygon* polygon1;
  const core::polygon* polygon2;
  core::edge           edge1;
  core::edge           edge2;
  violation_information(const core::polygon* polygon) {
    this->polygon1 = polygon;
  };
  violation_information(const core::polygon* polygon1,
                        const core::polygon* polygon2) {
    this->polygon1 = polygon1;
    this->polygon2 = polygon2;
  };
  violation_information(core::edge edge1, core::edge edge2) {
    this->edge1 = edge1;
    this->edge2 = edge2;
  };
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

template <typename N>
bool compare(N num1, N num2, rule_type ruletype) {
  if (ruletype == rule_type::enclosure or ruletype == rule_type::width) {
    return num1 > num2;
  } else {
    return num1 < num2;
  }
}

template <typename edge>
inline bool is_violation(edge      edge1,
                         edge      edge2,
                         int       threshold,
                         rule_type ruletype) {
  auto [start_point1, end_point1, distance1] = edge1;
  auto [start_point2, end_point2, distance2] = edge2;
  bool is_too_close = std::abs(distance1 - distance2) < threshold and distance1 !=distance2 ;
  bool is_right_side, is_projection_overlap;
  if (ruletype == rule_type::enclosure) {
    is_right_side =
        (end_point1 - start_point1) * (end_point2 - start_point2) > 0;
  } else {
    is_right_side =
        compare(distance1, distance2,ruletype)
            ? start_point1 > end_point1 and start_point2 < end_point2
            : start_point1 < end_point1 and start_point2 > end_point2;
  }
  is_projection_overlap =
      compare(distance1, distance2,ruletype)
          ? start_point1 > start_point2 and end_point1 < end_point2
          : start_point1 < start_point2 and end_point1 > end_point2;
  return is_too_close and is_projection_overlap and is_right_side;
}

void space_check_seq(odrc::core::database&               db,
                     std::vector<int>                    layers,
                     std::vector<int>                    without_layer,
                     int                                 threshold,
                     rule_type                           ruletype,
                     std::vector<violation_information>& vios);

void enclosing_check_seq(odrc::core::database&               db,
                         std::vector<int>                    layers,
                         std::vector<int>                    without_layer,
                         int                                 threshold,
                         rule_type                           ruletype,
                         std::vector<violation_information>& vios);
void area_check_seq(const odrc::core::database&         db,
                    int                                 layer,
                    int                                 threshold,
                    std::vector<violation_information>& vios);
void width_check_seq(const odrc::core::database&         db,
                     int                                 layer,
                     int                                 threshold,
                     rule_type                           ruletype,
                     std::vector<violation_information>& vios);

}  // namespace odrc