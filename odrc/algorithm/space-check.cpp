#include <iostream>
#include <odrc/algorithm/sequential_mode.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/engine.hpp>
#include <odrc/core/interval_tree.hpp>
namespace odrc {

void distance_check(odrc::core::database&               db,
                    int                                 layer,
                    std::vector<std::pair<int, int>>&   ovlpairs,
                    rule_type                           ruletype,
                    int                                 threshold,
                    std::vector<violation_information>& vios) {
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_h_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      check_distance(
          db.get_top_cell().cell_refs.at(f_cell).lower_edges.at(layer),
          db.get_top_cell().cell_refs.at(s_cell).upper_edges.at(layer),
          threshold, vios);
      check_distance(
          db.get_top_cell().cell_refs.at(f_cell).upper_edges.at(layer),
          db.get_top_cell().cell_refs.at(s_cell).lower_edges.at(layer),
          threshold, vios);
    }
  }
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_v_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      check_distance(
          db.get_top_cell().cell_refs.at(f_cell).left_edges.at(layer),
          db.get_top_cell().cell_refs.at(s_cell).right_edges.at(layer),
          threshold, vios);
      check_distance(
          db.get_top_cell().cell_refs.at(f_cell).right_edges.at(layer),
          db.get_top_cell().cell_refs.at(s_cell).left_edges.at(layer),
          threshold, vios);
    }
  }
}  

void distance_check(odrc::core::database&               db,
                    int                                 layer,
                    int                                 num,
                    rule_type                           ruletype,
                    int                                 threshold,
                    std::vector<violation_information>& vios) {
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_h_edge) {
    check_distance(db.cells.at(num).lower_edges.at(layer),
                   db.cells.at(num).upper_edges.at(layer), threshold, vios);
  }
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_v_edge) {
    check_distance(db.cells.at(num).left_edges.at(layer),
                   db.cells.at(num).right_edges.at(layer), threshold, vios);
  }
}

std::vector<std::pair<int, int>> get_ovlpairs(odrc::core::database& db,
                                              std::vector<int>&     layers,
                                              int                   threshold,
                                              std::vector<int>&     ids) {
  std::vector<std::pair<int, int>> ovlpairs;
  std::vector<event>               events;
  const auto&                      top_cell = db.get_top_cell();
  events.reserve(ids.size() * 2);
  for (auto i = 0UL; i < ids.size(); i++) {
    const auto& cell_ref = top_cell.cell_refs.at(ids[i]);
    int         idx      = db.get_cell_idx(cell_ref.cell_name);
    if (db.cells.at(idx).is_touching(layers)) {
      auto& mbr = cell_ref.cell_ref_mbr;
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max + threshold, int(ids[i])}, mbr.x_min,
                false, true});
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max + threshold, int(ids[i])}, mbr.x_max,
                false, false});
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

void space_check_seq(odrc::core::database&               db,
                     std::vector<int>                    layers,
                     std::vector<int>                    without_layer,
                     int                                 threshold,
                     rule_type                           ruletype,
                     std::vector<violation_information>& vios) {
  // get inter-cell violations
  auto rows = layout_partition(db, layers);
  for (auto row = 0UL; row < rows.size(); row++) {
    auto ovlpairs = get_ovlpairs(db, layers, threshold, rows[row]);
    distance_check(db, layers.front(), ovlpairs, ruletype, threshold, vios);
  }
  // get intra-cell violations
  std::map<int, std::vector<violation_information>> intra_vios;
  for (auto num = 0UL; num < db.cells.size(); num++) {
    if (db.cells.at(num).is_touching(layers) and num != db.get_top_cell_idx()) {
      intra_vios.emplace(num, std::vector<violation_information>());
      distance_check(db, layers.front(), num, ruletype, threshold,
                     intra_vios.at(num));
    }
  }
  get_ref_vios(db, intra_vios, vios);
}
}  // namespace odrc