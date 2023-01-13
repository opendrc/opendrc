#pragma once

#include <cassert>
#include <limits>
#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <odrc/core/edge.hpp>
#include <odrc/core/interval_tree.hpp>
namespace odrc {
using Intvl          = core::interval<int, int>;
using interval_pairs = std::vector<std::pair<int, int>>;
using edges          = std::vector<core::orthogonal_edge>;
enum class violation_type { area, distance, invalid };
struct event {
  Intvl intvl;
  int   y;
  bool  is_metal;
  bool  is_inevent;
};
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

// judge whether two edge is too close and overlapping
// the intercept of edge1 should be larger than the intercept of edge2
inline bool is_violation(const core::orthogonal_edge& edge1,
                         const core::orthogonal_edge& edge2,
                         int                          threshold) {
  auto [p1_start, p1_end, intercept1] = edge1;
  auto [p2_start, p2_end, intercept2] = edge2;
  if (std::abs(intercept1 - intercept2) < threshold) {
    return !(p1_start >= p2_end or p2_start >= p1_end);
  } else {
    return false;
  }
}

// transform intra-cell violations to inter-cell violations
inline void transform_vio(const violation&        vio,
                          const core::coord&      offset,
                          std::vector<violation>& vios) {
  core::edge edge1{{vio.distance.edge1.point1.x + offset.x,
                    vio.distance.edge1.point1.y + offset.y},
                   {vio.distance.edge1.point2.x + offset.x,
                    vio.distance.edge1.point2.y + offset.y}};
  core::edge edge2{{vio.distance.edge2.point1.x + offset.x,
                    vio.distance.edge2.point1.y + offset.y},
                   {vio.distance.edge2.point2.x + offset.x,
                    vio.distance.edge2.point2.y + offset.y}};
  vios.emplace_back(edge1, edge2);
}

inline void get_ref_vios(
    odrc::core::database&                        db,
    const std::map<int, std::vector<violation>>& intra_vios,
    std::vector<violation>&                      vios) {
  auto& cell_refs = db.get_top_cell().cell_refs;
  for (auto i = 0UL; i < cell_refs.size(); i++) {
    auto idx = db.get_cell_idx(cell_refs.at(i).cell_name);
    if (intra_vios.find(idx) != intra_vios.end()) {
      for (const auto& vio : intra_vios.at(idx)) {
        transform_vio(vio, cell_refs.at(i).ref_point, vios);
      }
    }
  }
}

inline void check_distance(const edges&            edges1,
                           const edges&            edges2,
                           int                     threshold,
                           std::vector<violation>& vios,
                           bool                    trans = true) {
  for (const auto& edge1 : edges1) {
    for (const auto& edge2 : edges2) {
      auto [p1_start, p1_end, intercept1] = edge1;
      auto [p2_start, p2_end, intercept2] = edge2;
      if (intercept1 > intercept2)
        if (is_violation(edge1, edge2, threshold)) {
          core::edge vio_edge1{core::coord{intercept1, p1_start, trans},
                               core::coord{intercept1, p1_end, trans}};
          core::edge vio_edge2{core::coord{intercept2, p2_start, trans},
                               core::coord{intercept2, p2_end, trans}};
          vios.emplace_back(violation{vio_edge1, vio_edge2});
        }
    }
  }
}

void space_check_seq(odrc::core::database&   db,
                     std::vector<int>        layers,
                     std::vector<int>        without_layer,
                     int                     threshold,
                     rule_type               ruletype,
                     std::vector<violation>& vios);

void enclosure_check_seq(odrc::core::database&   db,
                         const std::vector<int>& layers,
                         const std::vector<int>& without_layer,
                         const int               threshold,
                         const rule_type         ruletype,
                         std::vector<violation>& vios);

void area_check_seq(const odrc::core::database& db,
                    int                         layer,
                    int                         threshold,
                    std::vector<violation>&     vios);

void width_check_seq(odrc::core::database&   db,
                     int                     layer,
                     int                     threshold,
                     std::vector<violation>& vios);

}  // namespace odrc