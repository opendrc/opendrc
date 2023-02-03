#pragma once

#include <cassert>
#include <limits>

#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <odrc/core/edge.hpp>
#include <odrc/core/interval_tree.hpp>
#include <odrc/core/rule.hpp>

namespace odrc {
using Intvl = core::interval<int, int>;
template <typename V>
using OvlpPairs = std::vector<std::pair<V, V>>;
using edges     = std::vector<core::orthogonal_edge>;
struct event {
  Intvl intvl;
  int   y;
  bool  is_metal;
  bool  is_inevent;
};

// judge whether two edge is too close and overlapping
// the intercept of edge1 should be larger than the intercept of edge2
inline bool is_violation(const core::orthogonal_edge& edge1,
                         const core::orthogonal_edge& edge2,
                         int                          threshold) {
  auto [p1_start, p1_end, intercept1] = edge1;
  auto [p2_start, p2_end, intercept2] = edge2;
  if (std::abs(intercept1 - intercept2) < threshold) {
    return not(p1_start >= p2_end or p2_start >= p1_end);
  } else {
    return false;
  }
}

// transform intra-cell violations to inter-cell violations
inline void transform_vio(const core::violation&        vio,
                          const core::coord&            offset,
                          std::vector<core::violation>& vios) {
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
    odrc::core::database&                              db,
    const std::map<int, std::vector<core::violation>>& intra_vios,
    std::vector<core::violation>&                      vios) {
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

inline void check_distance(const edges&                  edges1,
                           const edges&                  edges2,
                           int                           threshold,
                           std::vector<core::violation>& vios,
                           bool                          is_horizontal = true) {
  for (const auto& edge1 : edges1) {
    for (const auto& edge2 : edges2) {
      auto [p1_start, p1_end, intercept1] = edge1;
      auto [p2_start, p2_end, intercept2] = edge2;
      if (intercept1 > intercept2)
        if (is_violation(edge1, edge2, threshold)) {
          core::edge vio_edge1{core::coord{intercept1, p1_start, is_horizontal},
                               core::coord{intercept1, p1_end, is_horizontal}};
          core::edge vio_edge2{core::coord{intercept2, p2_start, is_horizontal},
                               core::coord{intercept2, p2_end, is_horizontal}};
          vios.emplace_back(core::violation{vio_edge1, vio_edge2});
        }
    }
  }
}

void space_check_seq(odrc::core::database&         db,
                     std::vector<int>              layers,
                     std::vector<int>              without_layer,
                     int                           threshold,
                     core::rule_type               ruletype,
                     std::vector<core::violation>& vios);

void enclosure_check_seq(odrc::core::database&         db,
                         const std::vector<int>&       layers,
                         const std::vector<int>&       without_layer,
                         const int                     threshold,
                         const core::rule_type         ruletype,
                         std::vector<core::violation>& vios);

void area_check_seq(const odrc::core::database&   db,
                    int                           layer,
                    int                           threshold,
                    std::vector<core::violation>& vios);

void width_check_seq(odrc::core::database&         db,
                     int                           layer,
                     int                           threshold,
                     std::vector<core::violation>& vios);

}  // namespace odrc