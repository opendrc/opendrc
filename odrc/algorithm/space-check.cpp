#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/engine.hpp>
#include <odrc/core/interval_tree.hpp>
#include <odrc/core/structs.hpp>
namespace odrc {
using odrc::core::h_edge;
using odrc::core::v_edge;
using Intvl = core::interval<int, int>;

template <typename edge>
bool is_spacing_violation(edge f_edge, edge s_edge, int threshold) {
  auto [start_point1, end_point1, distance1] = f_edge;
  auto [start_point2, end_point2, distance2] = s_edge;
  bool is_too_close = std::abs(distance1 - distance2) < threshold;
  bool is_inversive =
      (end_point1 - start_point1) * (end_point2 - start_point2) < 0;
  bool is_projection_overlap =
      distance2 < distance1
          ? (start_point2 > start_point1 and end_point1 > end_point2)
          : (start_point1 > start_point2 and end_point2 > end_point1);
  return is_too_close and is_projection_overlap and is_inversive;
}

void distance_check(std::vector<std::pair<int, int>>& ovlpairs,
                    const core::edges&                edges,
                    rule_type                         ruletype,
                    int                               threshold,
                    std::vector<check_result>&        vios) {
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_h_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      const auto& f_edges = edges.h_edges.at(f_cell);
      const auto& s_edges = edges.h_edges.at(s_cell);
      for (const auto& f_edge : f_edges) {
        for (const auto& s_edge : s_edges) {
          auto [start_point1, end_point1, distance1] = f_edge;
          auto [start_point2, end_point2, distance2] = s_edge;
          bool is_violation = is_spacing_violation(f_edge, s_edge, threshold);
          if (is_violation) {
            vios.emplace_back(check_result{start_point1, distance1, end_point1,
                                           distance1, start_point2, distance2,
                                           end_point2, distance1, true});
          }
        }
      }
    }
  }
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_v_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      const auto& f_edges = edges.v_edges.at(f_cell);
      const auto& s_edges = edges.v_edges.at(s_cell);
      for (const auto& f_edge : f_edges) {
        for (const auto& s_edge : s_edges) {
          auto [start_point1, end_point1, distance1] = f_edge;
          auto [start_point2, end_point2, distance2] = s_edge;
          bool is_violation = is_spacing_violation(f_edge, s_edge, threshold);
          if (is_violation) {
            vios.emplace_back(check_result{distance1, start_point1, distance1,
                                           end_point1, distance2, start_point2,
                                           distance2, end_point2, true});
          }
        }
      }
    }
  }
}

/// @brief get overlapping pairs by interval tree
/// @param db database
/// @param layers the layer which two polygons should be in
/// @param rows  the number of divided layout row
/// @param row   the polygon number in the row
/// @return    the overlapping pairs
std::vector<std::pair<int, int>> get_ovlpairs(odrc::core::database& db,
                                              std::vector<int>&     layers,
                                              std::vector<int>&     ids) {
  std::vector<std::pair<int, int>> ovlpairs;
  std::vector<event>               events;
  const auto&                      top_cell = db.cells.back();
  events.reserve(ids.size() * 2);
  for (auto i = 0UL; i < ids.size(); i++) {
    const auto& cell_ref = top_cell.cell_refs.at(ids[i]);
    int         idx      = db.get_cell_idx(cell_ref.cell_name);
    if (db.cells.at(idx).is_touching(layers)) {
      auto& mbr = cell_ref.cell_ref_mbr;
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, int(i)}, mbr.x_min, false, true});
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, int(i)}, mbr.x_max, false, false});
    }
  }

  std::sort(events.begin(), events.end(), [](const auto& e1, const auto& e2) {
    return std::tie(e1.y, e2.is_inevent) < std::tie(e2.y, e1.is_inevent);
  });

  core::interval_tree<int, int> tree;
  ovlpairs.reserve(events.size() * 2);
  for (const auto& e : events) {
    if (e.is_inevent) {
      tree.get_intervals_overlapping_with(e.intvl, ovlpairs);
      tree.insert(e.intvl);
    } else {
      tree.remove(e.intvl);
    }
  }
  return ovlpairs;
}

/// @brief  sequence mode for spacing check between two polygons
/// @param db     database
/// @param layers the layer which two polygons should be in
/// @param without_layer  the layer which two polygons should be not in
/// @param threshold  the max spacing
/// @param ruletype some other limitation
/// @param vios return violations
void space_check_seq(odrc::core::database&      db,
                     std::vector<int>           layers,
                     std::vector<int>           without_layer,
                     int                        threshold,
                     rule_type                  ruletype,
                     std::vector<check_result>& vios) {
  auto        rows  = layout_partition(db, layers);
  const auto& edges = db.cell_edges[layers.front()];
  for (auto row = 0UL; row < rows.size(); row++) {
    auto ovlpairs = get_ovlpairs(db, layers, rows[row]);
    distance_check(ovlpairs, edges, ruletype, threshold, vios);
  }
}
}  // namespace odrc