#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/common_structs.hpp>
#include <odrc/core/engine.hpp>
#include <odrc/core/interval_tree.hpp>
namespace odrc {
using odrc::core::h_edge;
using odrc::core::v_edge;
using Intvl = core::interval<int, int>;

/// @brief detail horizontal edges check for overlapping polygons
/// @param ovlpairs overlapping polygon pairs
/// @param h_edges1 the horizontal edges of first polygon
/// @param h_edges2 the horizontal edges of second polygon
/// @param threshold the max spacing
/// @param vios  violation informations
void run_h_check(std::vector<std::pair<int, int>>& ovlpairs,
                 std::vector<std::vector<h_edge>>& hes,
                 int                               threshold,
                 std::vector<check_result>&        vios) {
  for (const auto& [f_poly, s_poly] : ovlpairs) {
    const auto& f_edges    = hes[f_poly];
    const auto& s_edges    = hes[s_poly];
    int         s_edge_min = s_edges.front().y - threshold;
    int         s_edge_max = s_edges.back().y + threshold;
    for (const auto& f_edge : f_edges) {
      if (f_edge.y < s_edge_min)
        continue;
      if (f_edge.y > s_edge_max)
        break;
      for (const auto& s_edge : s_edges) {
        if (s_edge.y < f_edge.y - threshold)
          continue;
        if (s_edge.y > f_edge.y + threshold)
          break;
        // space
        auto e11x         = f_edge.x1;
        auto e11y         = f_edge.y;
        auto e12x         = f_edge.x2;
        auto e12y         = f_edge.y;
        auto e21x         = s_edge.x1;
        auto e21y         = s_edge.y;
        auto e22x         = s_edge.x2;
        auto e22y         = s_edge.y;
        bool is_violation = false;
        if (e11y < e22y) {
          // e22 e21
          // e11 e12
          bool is_outside_to_outside = e11x < e12x and e21x > e22x;
          bool is_too_close          = e21y - e11y < threshold;
          bool is_projection_overlap = e21x < e11x and e12x < e22x;
          is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e12 e11
          // e21 e22
          bool is_outside_to_outside = e21x < e22x and e11x > e12x;
          bool is_too_close          = e11y - e21y < threshold;
          bool is_projection_overlap = e11x < e21x and e22x < e12x;
          is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        }
        if (is_violation) {
          vios.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x, e21y,
                                         e22x, e22y, true});
        }
      }
    }
  }
}

/// @brief detail vertical edges check for overlapping polygons
/// @param ovlpairs overlapping polygon pairs
/// @param v_edges1 the vertical edges of first polygon
/// @param v_edges2 the vertical edges of second polygon
/// @param threshold the max spacing
/// @param vios  violation informations
void run_v_check(std::vector<std::pair<int, int>>& ovlpairs,
                 std::vector<std::vector<v_edge>>& ves,
                 int                               threshold,
                 std::vector<check_result>&        vios) {
  for (const auto& [f_poly, s_poly] : ovlpairs) {
    const auto& f_edges    = ves[f_poly];
    const auto& s_edges    = ves[s_poly];
    int         s_edge_min = s_edges.front().x - threshold;
    int         s_edge_max = s_edges.back().x + threshold;
    for (const auto& f_edge : f_edges) {
      if (f_edge.x < s_edge_min)
        continue;
      if (f_edge.x > s_edge_max)
        break;
      for (const auto& s_edge : s_edges) {
        if (s_edge.x < f_edge.x - threshold)
          continue;
        if (s_edge.x > f_edge.x + threshold)
          break;
        // space
        auto e11y         = f_edge.y1;
        auto e11x         = f_edge.x;
        auto e12y         = f_edge.y2;
        auto e12x         = f_edge.x;
        auto e21y         = s_edge.y1;
        auto e21x         = s_edge.x;
        auto e22y         = s_edge.y2;
        auto e22x         = s_edge.x;
        bool is_violation = false;
        if (e11x < e22x) {
          // e22 e21
          // e11 e12
          bool is_outside_to_outside = e11y < e12y and e21y > e22y;
          bool is_too_close          = e12x - e11x < threshold;
          bool is_projection_overlap = e21y < e11y and e12y < e22y;
          is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e12 e11
          // e21 e22
          bool is_outside_to_outside = e21y < e22y and e11y > e12y;
          bool is_too_close          = e11x - e21x < threshold;
          bool is_projection_overlap = e11y < e21y and e22y < e12y;
          is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        }
        if (is_violation) {
          vios.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x, e21y,
                                         e22x, e22y, true});
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
std::vector<std::pair<int, int>> get_ovlpairs(
    odrc::core::database&          db,
    std::vector<int>&              layers,
    std::vector<std::vector<int>>& rows,
    int                            row) {
  std::vector<std::pair<int, int>> ovlpairs;
  std::vector<event>               events;
  const auto&                      top_cell = db.cells.back();
  events.reserve(rows[row].size() * 2);
  for (int i = 0; i < rows[row].size(); i++) {
    const auto& cell_ref = top_cell.cell_refs.at(rows[row][i]);
    int         idx      = db.get_cell_idx(cell_ref.cell_name);
    if (db.cells.at(idx).is_touching(layers)) {
      auto& mbr = db.edges[layers.front()].cell_ref_mbrs.at(idx);
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_min, false, true});
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_max, false, false});
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
  db.update_mbr(layers, without_layer);
  auto        rows  = layout_partition(db, layers);
  const auto& edges = db.edges[layers.front()];
  if (ruletype == rule_type::spacing_both) {
    for (int row = 0; row < rows.size(); row++) {
      std::vector<std::vector<h_edge>> h_edges;
      std::vector<std::vector<v_edge>> v_edges;
      for (const auto& i : rows[row]) {
        h_edges.emplace_back();
        v_edges.emplace_back();
        h_edges.back().insert(h_edges.back().end(), edges.h_edges.at(i).begin(),
                              edges.h_edges.at(i).end());
        v_edges.back().insert(v_edges.back().end(), edges.v_edges.at(i).begin(),
                              edges.v_edges.at(i).end());
      }
      auto ovlpairs = get_ovlpairs(db, layers, rows, row);
      run_h_check(ovlpairs, h_edges, threshold, vios);
      run_v_check(ovlpairs, v_edges, threshold, vios);
    }
  } else if (ruletype == rule_type::spacing_h_edge) {
    for (int row = 0; row < rows.size(); row++) {
      std::vector<std::vector<h_edge>> h_edges;
      for (const auto& i : rows[row]) {
        h_edges.emplace_back();
        h_edges.back().insert(h_edges.back().end(), edges.h_edges.at(i).begin(),
                              edges.h_edges.at(i).end());
      }
      auto ovlpairs = get_ovlpairs(db, layers, rows, row);
      run_h_check(ovlpairs, h_edges, threshold, vios);
    }
  } else if (ruletype == rule_type::spacing_v_edge) {
    for (int row = 0; row < rows.size(); row++) {
      std::vector<std::vector<v_edge>> v_edges;
      for (const auto& i : rows[row]) {
        v_edges.emplace_back();
        v_edges.back().insert(v_edges.back().end(), edges.v_edges.at(i).begin(),
                              edges.v_edges.at(i).end());
      }
      auto ovlpairs = get_ovlpairs(db, layers, rows, row);
      run_v_check(ovlpairs, v_edges, threshold, vios);
    }
  }
  db.erase_mbr();
}
}  // namespace odrc