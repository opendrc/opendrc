#include <odrc/algorithm/sequential_mode.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/engine.hpp>
#include <odrc/core/interval_tree.hpp>
#include <odrc/utility/logger.hpp>
#include <odrc/utility/timer.hpp>

namespace odrc {

// inter-cell violation check
void _check(odrc::core::database&         db,
            int                           layer,
            interval_pairs&               ovlpairs,
            core::rule_type               ruletype,
            int                           threshold,
            std::vector<core::violation>& vios) {
  auto& cell_refs = db.get_top_cell().cell_refs;
  if (ruletype == core::rule_type::spacing_both or
      ruletype == core::rule_type::spacing_h_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      check_distance(cell_refs.at(f_cell).lower_edges.at(layer),
                     cell_refs.at(s_cell).upper_edges.at(layer), threshold,
                     vios);
      check_distance(cell_refs.at(s_cell).lower_edges.at(layer),
                     cell_refs.at(f_cell).upper_edges.at(layer), threshold,
                     vios);
    }
  }
  if (ruletype == core::rule_type::spacing_both or
      ruletype == core::rule_type::spacing_v_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      check_distance(cell_refs.at(f_cell).left_edges.at(layer),
                     cell_refs.at(s_cell).right_edges.at(layer), threshold,
                     vios, false);
      check_distance(cell_refs.at(s_cell).left_edges.at(layer),
                     cell_refs.at(f_cell).right_edges.at(layer), threshold,
                     vios, false);
    }
  }
}

// intra-cell violation check
void _check(odrc::core::database&         db,
            int                           layer,
            int                           i,
            core::rule_type               ruletype,
            int                           threshold,
            std::vector<core::violation>& vios) {
  if (ruletype == core::rule_type::spacing_both or
      ruletype == core::rule_type::spacing_h_edge) {
    check_distance(db.cells.at(i).lower_edges.at(layer),
                   db.cells.at(i).upper_edges.at(layer), threshold, vios);
  }
  if (ruletype == core::rule_type::spacing_both or
      ruletype == core::rule_type::spacing_v_edge) {
    check_distance(db.cells.at(i).left_edges.at(layer),
                   db.cells.at(i).right_edges.at(layer), threshold, vios);
  }
}

interval_pairs get_ovlpairs(odrc::core::database& db,
                            std::vector<int>&     layers,
                            int                   threshold,
                            std::vector<int>&     ids) {
  interval_pairs     ovlpairs;
  std::vector<event> events;
  auto&              cell_refs = db.get_top_cell().cell_refs;
  events.reserve(ids.size() * 2);
  for (auto i = 0UL; i < ids.size(); i++) {
    const auto& cell_ref = cell_refs.at(ids[i]);
    int         idx      = db.get_cell_idx(cell_ref.cell_name);
    if (db.cells.at(idx).is_touching(layers)) {
      auto& mbr = cell_ref.cell_ref_mbr;
      /// utilize threshold to expand mbr
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max + threshold - 1, int(ids[i])},
                mbr.x_min, false, true});
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max + threshold - 1, int(ids[i])},
                mbr.x_max + threshold - 1, false, false});
    }
  }

  std::sort(events.begin(), events.end(), [](const auto& e1, const auto& e2) {
    return std::tie(e1.y, e2.is_inevent) < std::tie(e2.y, e1.is_inevent);
  });

  core::interval_tree<int, int> tree;
  ovlpairs.reserve(events.size() * 2);
  for (const auto& e : events) {
    if (e.is_inevent) {
      tree.get_intervals_pairs(e.intvl, ovlpairs);
      tree.insert(e.intvl);
    } else {
      tree.remove(e.intvl);
    }
  }
  return ovlpairs;
}

void space_check_seq(odrc::core::database&         db,
                     std::vector<int>              layers,
                     std::vector<int>              without_layer,
                     int                           threshold,
                     core::rule_type               ruletype,
                     std::vector<core::violation>& vios) {
  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  odrc::util::timer  space_check("space_check", logger);
  // get inter-cell violations
  auto rows = layout_partition(db, layers, threshold);
  space_check.start();
  for (auto row = 0UL; row < rows.size(); row++) {
    auto ovlpairs = get_ovlpairs(db, layers, threshold, rows[row]);
    _check(db, layers.front(), ovlpairs, ruletype, threshold, vios);
  }
  space_check.pause();
  // get intra-cell violations
  std::map<int, std::vector<core::violation>> intra_vios;
  for (auto idx = 0UL; idx < db.cells.size(); idx++) {
    if (db.cells.at(idx).is_touching(layers) and idx != db.top_cell_id) {
      intra_vios.emplace(idx, std::vector<core::violation>());
      _check(db, layers.front(), idx, ruletype, threshold, intra_vios.at(idx));
    }
  }
  get_ref_vios(db, intra_vios, vios);
}
}  // namespace odrc