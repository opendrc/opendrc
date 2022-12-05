#pragma once

#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <vector>

namespace odrc::core {
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

struct rule {
  int                 rule_num;
  int                 layer;
  std::vector<int>    with_layer;
  std::vector<int>    without_layer;
  std::pair<int, int> region;
  rule_type           ruletype;
  object              obj   = object::both;
  sramdrc_set         s_set = sramdrc_set::dft;
};

class engine {
 public:
  std::vector<std::pair<int, int>> vlts;
  //<rule number, polgon/cell number>
  std::vector<std::pair<int, std::pair<int, int>>> vlt_paires;
  //<rule number,<polygons/cells number pair>>

  void add_rules(std::vector<int> value){};
  void check(odrc::core::database& db){};

  engine& polygons() {
    rules.emplace_back();
    rules.back().rule_num = rule_num;
    rules.back().obj      = object::polygon;
    return *this;
  }
  engine& layer(int layer) {
    rules.emplace_back();
    rules.back().rule_num = rule_num;
    rules.back().layer    = layer;
    return *this;
  }
  engine& width() {
    rules.back().ruletype = rule_type::width;
    return *this;
  }

  int is_rectilinear() {
    rules.back().ruletype = rule_type::aux_is_rectilinear;
    return -1;
  }
  int greater_than(int min) {
    rules.back().region = std::make_pair(min, 2147483647);
    return -1;
  }

 private:
  unsigned int      rule_num = 0;
  std::vector<rule> rules;
};

}  // namespace odrc::core