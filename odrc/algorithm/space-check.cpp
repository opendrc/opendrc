#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/engine.hpp>
#include <odrc/core/interval_tree.hpp>

namespace odrc {

using odrc::core::h_edge;
using odrc::core::v_edge;
using coord        = odrc::core::coord;
using polygon      = odrc::core::polygon;
using cell_ref     = odrc::core::cell_ref;
using check_result = odrc::check_result;
using Intvl        = core::interval<int, int>;


struct event {
  Intvl intvl;
  int   y;
  bool  is_polygon;
  bool  is_inevent;
};

void run_h_check(std::vector<std::pair<int, int>>& ovlpairs,
                 h_edge*                           hes,
                 int*                              hidx,
                 int                               threshold,
                 std::vector<check_result>&        vios) {
  for (const auto& pair : ovlpairs) {
    int c1    = pair.first;
    int c2    = pair.second;
    int h1s   = hidx[c1];
    int h1e   = hidx[c1 + 1];
    int h2s   = hidx[c2];
    int h2e   = hidx[c2 + 1];
    int h1min = hes[h1s].y;
    int h1max = hes[(h1e - 1)].y;
    int h2min = hes[h2s].y - threshold;
    int h2max = hes[(h2e - 1)].y + threshold;
    for (auto p1 = h1s; p1 < h1e; ++p1) {
      int p1y = hes[p1].y;
      if (hes[p1].y < h2min)
        continue;
      if (hes[p1].y > h2max)
        break;
      for (auto p2 = h2s; p2 < h2e; ++p2) {
        if (hes[p2].y < p1y - threshold)
          continue;
        if (hes[p2].y > p1y + threshold)
          break;
        // space
        auto e11x         = hes[p1].x1;
        auto e11y         = hes[p1].y;
        auto e12x         = hes[p1].x2;
        auto e12y         = hes[p1].y;
        auto e21x         = hes[p2].x1;
        auto e21y         = hes[p2].y;
        auto e22x         = hes[p2].x2;
        auto e22y         = hes[p2].y;
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
void run_v_check(std::vector<std::pair<int, int>>& ovlpairs,
                 v_edge*                           ves,
                 int*                              vidx,
                 int                               threshold,
                 std::vector<check_result>&        vios) {
  for (const auto& pair : ovlpairs) {
    int c1    = pair.first;
    int c2    = pair.second;
    int h1s   = vidx[c1];
    int h1e   = vidx[c1 + 1];
    int h2s   = vidx[c2];
    int h2e   = vidx[c2 + 1];
    int h1min = ves[h1s].x;
    int h1max = ves[(h1e - 1)].x;
    int h2min = ves[h2s].x - threshold;
    int h2max = ves[(h2e - 1)].x + threshold;
    for (auto p1 = h1s; p1 < h1e; ++p1) {
      int p1y = ves[p1].x;
      if (ves[p1].x < h2min)
        continue;
      if (ves[p1].x > h2max)
        break;
      for (auto p2 = h2s; p2 < h2e; ++p2) {
        if (ves[p2].x < p1y - threshold)
          continue;
        if (ves[p2].x > p1y + threshold)
          break;
        // space
        auto e11y         = ves[p1].y1;
        auto e11x         = ves[p1].x;
        auto e12y         = ves[p1].y2;
        auto e12x         = ves[p1].x;
        auto e21y         = ves[p2].y1;
        auto e21x         = ves[p2].x;
        auto e22y         = ves[p2].y2;
        auto e22x         = ves[p2].x;
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

std::vector<std::pair<int, int>> get_ovlpairs(
    const odrc::core::database&    db,
    std::vector<int>&              layers,
    std::vector<std::vector<int>>& rows,
    int                            row) {
  std::vector<std::pair<int, int>> ovlpairs;
  std::vector<event>               events;
  const auto&                      top_cell = db.cells.back();
  events.reserve(rows[row].size() * 2);
  for (int i = 0; i < rows[row].size(); i++) {
    const auto& cell_ref = top_cell.cell_refs.at(rows[row][i]);
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (!the_cell.is_touching(layers.front()) and
        !the_cell.is_touching(layers.back())) {
      continue;
    }
    events.emplace_back(event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3], i},
                              cell_ref.mbr[0], false, true});
    events.emplace_back(event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3], i},
                              cell_ref.mbr[1], false, false});
  }
  {
    std::sort(events.begin(), events.end(), [](const auto& e1, const auto& e2) {
      return e1.y == e2.y ? (e1.is_inevent and !e2.is_inevent) : e1.y < e2.y;
    });
  }
  core::interval_tree<int, int> tree;
  ovlpairs.reserve(events.size() * 2);
  for (int i = 0; i < events.size(); ++i) {
    const auto& e = events[i];
    if (e.is_inevent) {
      tree.get_intervals_overlapping_with(e.intvl, ovlpairs);
      tree.insert(e.intvl);
    } else {
      tree.remove(e.intvl);
    }
  }
  return ovlpairs;
}

void space_check_seq(const odrc::core::database& db,
                     std::vector<int>            layers,
                     int                         threshold,
                     rule_type                   ruletype,
                     std::vector<check_result>&  vios) {
  auto        rows     = layout_partition(db, layers);
  const auto& top_cell = db.cells.back();
  if (ruletype == rule_type::spacing_both) {
    for (int row = 0; row < rows.size(); row++) {
      std::vector<h_edge> hes;
      std::vector<v_edge> ves;
      std::vector<int>    hidx;
      std::vector<int>    vidx;
      for (int i = 0; i < rows[row].size(); i++) {
        int cr = rows[row][i];
        hidx.emplace_back(hes.size());
        vidx.emplace_back(ves.size());
        hes.insert(hes.end(), top_cell.cell_refs.at(cr).h_edges1.begin(),
                   top_cell.cell_refs.at(cr).h_edges1.end());
        ves.insert(ves.end(), top_cell.cell_refs.at(cr).v_edges1.begin(),
                   top_cell.cell_refs.at(cr).v_edges1.end());
      }
      hidx.emplace_back(hes.size());
      vidx.emplace_back(ves.size());
      auto ovlpairs = get_ovlpairs(db, layers, rows, row);
      run_h_check(ovlpairs, hes.data(), hidx.data(), threshold, vios);
      run_v_check(ovlpairs, ves.data(), vidx.data(), threshold, vios);
    }
  } else if (ruletype == rule_type::spacing_h_edge) {
    for (int row = 0; row < rows.size(); row++) {
      std::vector<h_edge> hes;
      std::vector<int>    hidx;
      for (int i = 0; i < rows[row].size(); i++) {
        int cr = rows[row][i];
        hidx.emplace_back(hes.size());
        hes.insert(hes.end(), top_cell.cell_refs.at(cr).h_edges1.begin(),
                   top_cell.cell_refs.at(cr).h_edges1.end());
      }
      hidx.emplace_back(hes.size());
      auto ovlpairs = get_ovlpairs(db, layers, rows, row);
      run_h_check(ovlpairs, hes.data(), hidx.data(), threshold, vios);
    }
  } else if (ruletype == rule_type::spacing_v_edge) {
    for (int row = 0; row < rows.size(); row++) {
      std::vector<v_edge> ves;
      std::vector<int>    vidx;
      for (int i = 0; i < rows[row].size(); i++) {
        int cr = rows[row][i];
        vidx.emplace_back(ves.size());
        ves.insert(ves.end(), top_cell.cell_refs.at(cr).v_edges1.begin(),
                   top_cell.cell_refs.at(cr).v_edges1.end());
      }
      vidx.emplace_back(ves.size());
      auto ovlpairs = get_ovlpairs(db, layers, rows, row);
      run_v_check(ovlpairs, ves.data(), vidx.data(), threshold, vios);
    }
  }
}

}  // namespace odrc