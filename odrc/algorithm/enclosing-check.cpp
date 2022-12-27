#include <odrc/algorithm/sequential_mode.hpp>

#include <algorithm>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/database.hpp>
#include <odrc/core/interval_tree.hpp>

namespace odrc {

using Intvl = core::interval<int, int>;

void distance_check(odrc::core::database& db,
                    std::vector<int>      layers,
                    rule_type                           ruletype,
                    std::vector<std::pair<int, int>>&   ovlpairs,
                    int                                 threshold,
                    std::vector<violation_information>& vios) {
  for (const auto& [f_cell, s_cell] : ovlpairs) {
    auto& edges1 =
        db.cells.back().cell_refs.at(f_cell).h_edges.at(layers.front());
    auto& edges2 =
        db.cells.back().cell_refs.at(s_cell).h_edges.at(layers.back());
    for (const auto& edge1 : edges1) {
      for (const auto& edge2 : edges2) {
        auto [start_point1, end_point1, distance1] = edge1;
        auto [start_point2, end_point2, distance2] = edge2;
        bool is_violation = is_spacing_violation(edge1, edge2, threshold,ruletype);
        if (is_violation) {
          vios.emplace_back(violation_information{
              core::edge{start_point1, distance1, end_point1, distance1},
              core::edge{start_point2, distance2, end_point2, distance1}});
        }
      }
    }
  }
  for (const auto& [f_cell, s_cell] : ovlpairs) {
    auto& edges1 =
        db.cells.back().cell_refs.at(f_cell).v_edges.at(layers.front());
    auto& edges2 =
        db.cells.back().cell_refs.at(s_cell).v_edges.at(layers.back());
    for (const auto& edge1 : edges1) {
      for (const auto& edge2 : edges2) {
        auto [start_point1, end_point1, distance1] = edge1;
        auto [start_point2, end_point2, distance2] = edge2;
        bool is_violation = is_spacing_violation(edge1, edge2, threshold,ruletype);
        if (is_violation) {
          vios.emplace_back(violation_information{
              core::edge{distance1, start_point1, distance1, end_point1},
              core::edge{distance2, start_point2, distance2, end_point2}});
        }
      }
    }
  }
}

std::vector<std::pair<int, int>> get_enclosing_ovlpairs(
    odrc::core::database& db,
    std::vector<int>&     layers,
    std::vector<int>&     ids) {
  std::vector<std::pair<int, int>> ovlpairs;
  std::vector<event>               events;
  const auto&                      top_cell = db.cells.back();
  events.reserve(ids.size() * 2);
  for (int i = 0; i < int(ids.size()); i++) {
    const auto& cell_ref = top_cell.cell_refs.at(ids[i]);
    int         idx      = db.get_cell_idx(cell_ref.cell_name);
    if (db.cells.at(idx).is_touching(layers.back())) {  // layer 2 is via
      auto& mbr = cell_ref.cell_ref_mbr;
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_min, false, true});
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_max, false, false});
    } else if (db.cells.at(idx).is_touching(
                   layers.front())) {  // layer 1 is metal
      auto& mbr = cell_ref.cell_ref_mbr;
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_min, false, true});
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_max, false, false});
    } else {
      continue;
    }
  }
  {
    std::sort(events.begin(), events.end(), [](const auto& e1, const auto& e2) {
      return std::tie(e1.y, e2.is_inevent) < std::tie(e2.y, e1.is_inevent);
    });
  }
  core::interval_tree<int, int> tree_V;
  core::interval_tree<int, int> tree_M;
  ovlpairs.reserve(events.size() * 2);
  for (auto i = 0UL; i < events.size(); ++i) {
    const auto& e = events[i];
    if (e.is_polygon) {  // metal
      if (e.is_inevent) {
        tree_V.get_intervals_overlapping_with(e.intvl, ovlpairs);
        tree_M.insert(e.intvl);
      } else {
        tree_M.remove(e.intvl);
      }
    } else {
      if (e.is_inevent) {
        tree_M.get_intervals_overlapping_with(e.intvl, ovlpairs);
        tree_V.insert(e.intvl);
      } else {
        tree_V.remove(e.intvl);
      }
    }
  }
  return ovlpairs;
}

void enclosing_check_seq(odrc::core::database&               db,
                         std::vector<int>                    layers,
                         std::vector<int>                    without_layer,
                         int                                 threshold,
                         rule_type                           ruletype,
                         std::vector<violation_information>& vios) {
  auto rows = layout_partition(db, layers);
  // get edges from cell
  for (auto row = 0UL; row < rows.size(); row++) {
    auto ovlpairs = get_enclosing_ovlpairs(db, layers, rows[row]);
    distance_check(db, layers,ruletype, ovlpairs, threshold, vios);
  }
}

}  // namespace odrc