#include <odrc/algorithm/sequential_mode.hpp>

#include <algorithm>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/engine.hpp>
#include <odrc/core/interval_tree.hpp>
namespace odrc {
using odrc::core::h_edge;
using odrc::core::v_edge;

void distance_check(odrc::core::database&               db,
                    std::vector<int>&                   layers,
                    std::vector<std::pair<int, int>>&   ovlpairs,
                    rule_type                           ruletype,
                    int                                 threshold,
                    std::vector<violation_information>& vios) {
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_h_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      const auto& edges1 =
          db.cells.back().cell_refs.at(f_cell).h_edges.at(layers.front());
      const auto& edges2 =
          db.cells.back().cell_refs.at(s_cell).h_edges.at(layers.front());
      for (const auto& f_edge : edges1) {
        for (const auto& s_edge : edges2) {
          auto [start_point1, end_point1, distance1] = f_edge;
          auto [start_point2, end_point2, distance2] = s_edge;
          bool is_violation = is_spacing_violation(f_edge, s_edge, threshold);
          if (is_violation) {
            vios.emplace_back(violation_information{
                core::edge{start_point1, distance1, end_point1, distance1},
                core::edge{start_point2, distance2, end_point2, distance1}});
          }
        }
      }
    }
  }
  if (ruletype == rule_type::spacing_both or
      ruletype == rule_type::spacing_v_edge) {
    for (const auto& [f_cell, s_cell] : ovlpairs) {
      const auto& edges1 =
          db.cells.back().cell_refs.at(f_cell).h_edges.at(layers.front());
      const auto& edges2 =
          db.cells.back().cell_refs.at(s_cell).h_edges.at(layers.front());
      for (const auto& f_edge : edges1) {
        for (const auto& s_edge : edges2) {
          auto [start_point1, end_point1, distance1] = f_edge;
          auto [start_point2, end_point2, distance2] = s_edge;
          bool is_violation = is_spacing_violation(f_edge, s_edge, threshold);
          if (is_violation) {
            vios.emplace_back(violation_information{
                core::edge{distance1, start_point1, distance1, end_point1},
                core::edge{distance2, start_point2, distance2, end_point2}});
          }
        }
      }
    }
  }
}  // namespace odrc

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
      // tree.get_intervals_overlapping_with(e.intvl, ovlpairs);
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
  auto rows = layout_partition(db, layers);
  for (auto row = 0UL; row < rows.size(); row++) {
    auto ovlpairs = get_ovlpairs(db, layers, rows[row]);
    distance_check(db, layers, ovlpairs, ruletype, threshold, vios);
  }
}
}  // namespace odrc