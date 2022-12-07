#pragma once

#include <odrc/core/database.hpp>
namespace odrc {
struct check_result {
  int  e11x;
  int  e11y;
  int  e12x;
  int  e12y;
  int  e21x;
  int  e21y;
  int  e22x;
  int  e22y;
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
  aux_is_rectilinear
};
void space_check_seq(const odrc::core::database& db,
                     std::vector<int>           layers,
                     int                        threshold,
                     rule_type                  ruletype,
                     std::vector<check_result>& vios);

void space_check_pal(const odrc::core::database& db,
                     int                         layer1,
                     int                         layer2,
                     int                         threshold,
                     std::vector<check_result>&  vios);
}  // namespace odrc