#include <odrc/algorithm/enclosing-check.hpp>

#include <algorithm>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/common_structs.hpp>
#include <odrc/core/database.hpp>
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
                 std::vector<std::vector<h_edge>>& h_edges1,
                 std::vector<std::vector<h_edge>>& h_edges2,
                 int                               threshold,
                 std::vector<check_result>&        vios) {
  for (const auto& [f_poly, s_poly] : ovlpairs) {
    const auto& f_edges    = h_edges2[f_poly];
    const auto& s_edges    = h_edges1[s_poly];
    int         s_edge_min = s_edges.front().y - threshold;
    int         s_edge_max = s_edges.back().y + threshold;
    // std::cout << "111" << std::endl;
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
          // e12 e11
          bool is_inside_to_outside  = e11x > e12x and e21x > e22x;
          bool is_too_close          = e21y - e11y < threshold;
          bool is_projection_overlap = e22x > e12x and e21x < e11x;
          is_violation =
              is_inside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e11 e12
          // e21 e22
          bool is_inside_to_outside  = e11x < e12x and e21x < e22x;
          bool is_too_close          = e11y - e21y < threshold;
          bool is_projection_overlap = e12x < e22x and e21x < e11x;
          is_violation =
              is_inside_to_outside and is_too_close and is_projection_overlap;
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
                 std::vector<std::vector<v_edge>>& v_edges1,
                 std::vector<std::vector<v_edge>>& v_edges2,
                 int                               threshold,
                 std::vector<check_result>&        vios) {
  for (const auto& [f_poly, s_poly] : ovlpairs) {
    const auto& f_edges    = v_edges2[f_poly];
    const auto& s_edges    = v_edges1[s_poly];
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
          // e12 e22
          // e11 e21
          bool is_inside_to_outside  = e11y < e12y and e22y > e21y;
          bool is_too_close          = e12x - e11x < threshold;
          bool is_projection_overlap = e21y > e11y and e12y > e22y;
          is_violation =
              is_inside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e21 e11
          // e22 e12
          bool is_inside_to_outside  = e21y > e22y and e11y > e12y;
          bool is_too_close          = e11x - e21x < threshold;
          bool is_projection_overlap = e12y < e22y and e21y < e11y;
          is_violation =
              is_inside_to_outside and is_too_close and is_projection_overlap;
        }
        if (is_violation) {
          vios.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x, e21y,
                                         e22x, e22y, true});
        }
      }
    }
  }
}

/// @brief get overlapping pairs by two interval trees
/// @param db database
/// @param layers the layer which two polygons should be in
/// @param rows  the number of divided layout row
/// @param row   the polygon number in the row
/// @return    the overlapping pairs
std::vector<std::pair<int, int>> get_enclosing_ovlpairs(
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
    if (db.cells.at(idx).is_touching(layers.back())) {  // layer 2 is via
      auto& mbr = db.edges[layers.back()].cell_ref_mbrs.at(idx);
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_min, false, true});
      events.emplace_back(
          event{Intvl{mbr.y_min, mbr.y_max, i}, mbr.x_max, false, false});
    } else if (db.cells.at(idx).is_touching(
                   layers.front())) {  // layer 1 is metal
      auto& mbr = db.edges[layers.front()].cell_ref_mbrs.at(idx);
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
  for (int i = 0; i < events.size(); ++i) {
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

/// @brief  sequence mode for spacing check between two enclosing polygons
/// @param db     database
/// @param layers the layer which two polygons should be in
/// @param without_layer  the layer which two polygons should be not in
/// @param threshold  the max spacing
/// @param ruletype some other limitation
/// @param vios return violations
void enclosing_check_seq(odrc::core::database&      db,
                         std::vector<int>           layers,
                         std::vector<int>           without_layer,
                         int                        threshold,
                         rule_type                  ruletype,
                         std::vector<check_result>& vios) {
  // layers.front() should be metal layer and layers.back() should be via layer
  db.update_mbr(layers, without_layer);
  auto        rows   = layout_partition(db, layers);
  const auto& edges1 = db.edges[layers.front()];
  const auto& edges2 = db.edges[layers.back()];
  // get edges from cell
  for (int row = 0; row < rows.size(); row++) {
    std::vector<std::vector<h_edge>> h_edges1;
    std::vector<std::vector<v_edge>> v_edges1;
    std::vector<std::vector<h_edge>> h_edges2;
    std::vector<std::vector<v_edge>> v_edges2;
    for (const auto& i : rows[row]) {
      h_edges1.emplace_back();
      v_edges1.emplace_back();
      h_edges2.emplace_back();
      v_edges2.emplace_back();
      h_edges1.back().insert(h_edges1.back().end(),
                             edges1.h_edges.at(i).begin(),
                             edges1.h_edges.at(i).end());
      v_edges1.back().insert(v_edges1.back().end(),
                             edges2.v_edges.at(i).begin(),
                             edges2.v_edges.at(i).end());
      h_edges2.back().insert(h_edges2.back().end(),
                             edges1.h_edges.at(i).begin(),
                             edges1.h_edges.at(i).end());
      v_edges2.back().insert(v_edges2.back().end(),
                             edges2.v_edges.at(i).begin(),
                             edges2.v_edges.at(i).end());
    }
    auto ovlpairs = get_enclosing_ovlpairs(db, layers, rows, row);
    run_h_check(ovlpairs, h_edges1, h_edges2, threshold, vios);
    run_v_check(ovlpairs, v_edges1, v_edges2, threshold, vios);
  }
  db.erase_mbr();
}

}  // namespace odrc