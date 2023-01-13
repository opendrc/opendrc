#pragma once

#include <cassert>
#include <vector>

#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <odrc/core/edge.hpp>

namespace odrc::core {

enum class violation_type { area, distance, invalid };

struct violation {
  violation_type type;
  union {
    struct {
      const core::polygon* poly;
    } area;
    struct {
      core::edge edge1;
      core::edge edge2;
    } distance;
  };
  violation(const core::polygon* polygon) { this->area.poly = polygon; };
  violation(const core::edge edge1, const core::edge edge2) {
    this->distance.edge1 = edge1;
    this->distance.edge2 = edge2;
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

enum class object {
  polygon,
  cell,
  both,
};

enum class sramdrc_set {
  dft              = 0,  // default
  interac_SRAMDRAC = 1,  // interact with the layer SRAMDRC
  outside_SRAMDRAC = 2,  // outside the layer SRAMDRC
};

enum class mode { sequential, parallel };

struct rule {
  int                 rule_num;
  std::vector<int>    layer;
  std::vector<int>    with_layer;
  std::vector<int>    without_layer;
  std::pair<int, int> region;
  rule_type           ruletype;
  object              obj   = object::both;
  sramdrc_set         s_set = sramdrc_set::dft;
};

}  // namespace odrc::core