#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/database.hpp>
#include <odrc/core/interval_tree.hpp>
#include <odrc/utility/logger.hpp>
#include <odrc/utility/timer.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace odrc {

using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;
using odrc::core::h_edge;
using odrc::core::v_edge;
using Intvl = core::interval<int, int>;
struct event {
  Intvl intvl;
  int   y;
  bool  is_polygon;
  bool  is_inevent;
};

void run_check_hedge(std::vector<std::pair<int, int>>& ovlpairs,
                     h_edge*                           hes1,
                     h_edge*                           hes2,
                     int*                              hidx1,
                     int*                              hidx2,
                     int                               threshold,
                     std::vector<check_result>&        vios) {
  for (const auto& pair : ovlpairs) {
    int c1    = pair.first;
    int c2    = pair.second;
    int h1s   = hidx1[c1];
    int h1e   = hidx1[c1 + 1];
    int h2s   = hidx2[c2];
    int h2e   = hidx2[c2 + 1];
    int h1min = hes1[h1s].y;
    int h1max = hes1[(h1e - 1)].y;
    int h2min = hes2[h2s].y - threshold;
    int h2max = hes2[(h2e - 1)].y + threshold;

    for (auto p1 = h1s; p1 < h1e; ++p1) {
      int p1y = hes1[p1].y;
      if (hes1[p1].y < h2min)
        continue;
      if (hes1[p1].y > h2max)
        break;
      for (auto p2 = h2s; p2 < h2e; ++p2) {
        if (hes2[p2].y < p1y - threshold)
          continue;
        if (hes2[p2].y > p1y + threshold)
          break;
        // space
        auto e11x         = hes1[p1].x1;
        auto e11y         = hes1[p1].y;
        auto e12x         = hes1[p1].x2;
        auto e12y         = hes1[p1].y;
        auto e21x         = hes2[p2].x1;
        auto e21y         = hes2[p2].y;
        auto e22x         = hes2[p2].x2;
        auto e22y         = hes2[p2].y;
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
void run_check_vedge(std::vector<std::pair<int, int>>& ovlpairs,
                     v_edge*                           ves1,
                     v_edge*                           ves2,
                     int*                              vidx1,
                     int*                              vidx2,
                     int                               threshold,
                     std::vector<check_result>&        vios) {
  for (const auto& pair : ovlpairs) {
    int c1    = pair.first;
    int c2    = pair.second;
    int h1s   = vidx1[c1];
    int h1e   = vidx1[c1 + 1];
    int h2s   = vidx2[c2];
    int h2e   = vidx2[c2 + 1];
    int h1min = ves1[h1s].x;
    int h1max = ves1[(h1e - 1)].x;
    int h2min = ves2[h2s].x - threshold;
    int h2max = ves2[(h2e - 1)].x + threshold;

    for (auto p1 = h1s; p1 < h1e; ++p1) {
      int p1y = ves1[p1].x;
      if (ves1[p1].x < h2min)
        continue;
      if (ves1[p1].x > h2max)
        break;
      for (auto p2 = h2s; p2 < h2e; ++p2) {
        if (ves2[p2].x < p1y - threshold)
          continue;
        if (ves2[p2].x > p1y + threshold)
          break;
        // space
        auto e11y         = ves1[p1].y1;
        auto e11x         = ves1[p1].x;
        auto e12y         = ves1[p1].y2;
        auto e12x         = ves1[p1].x;
        auto e21y         = ves2[p2].y1;
        auto e21x         = ves2[p2].x;
        auto e22y         = ves2[p2].y2;
        auto e22x         = ves2[p2].x;
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

void enclosing_check_seq(const odrc::core::database& db,
                         std::vector<int>            layers,
                         int                         threshold,
                         rule_type                   ruletype,
                         std::vector<check_result>&  vios) {
  auto        rows     = layout_partition(db, layers);
  const auto& top_cell = db.cells.back();

  for (int row = 0; row < rows.size(); row++) {
    std::vector<h_edge> hes1;
    std::vector<h_edge> hes2;
    std::vector<v_edge> ves1;
    std::vector<v_edge> ves2;
    std::vector<int>    hidx1;
    std::vector<int>    hidx2;
    std::vector<int>    vidx1;
    std::vector<int>    vidx2;
    for (int i = 0; i < rows[row].size(); i++) {
      int cr = rows[row][i];
      hidx1.emplace_back(hes1.size());
      vidx1.emplace_back(ves1.size());
      hes1.insert(hes1.end(), top_cell.cell_refs.at(cr).h_edges1.begin(),
                  top_cell.cell_refs.at(cr).h_edges1.end());
      ves1.insert(ves1.end(), top_cell.cell_refs.at(cr).v_edges1.begin(),
                  top_cell.cell_refs.at(cr).v_edges1.end());
      hidx2.emplace_back(hes2.size());
      vidx2.emplace_back(ves2.size());
      hes2.insert(hes2.end(), top_cell.cell_refs.at(cr).h_edges2.begin(),
                  top_cell.cell_refs.at(cr).h_edges2.end());
      ves2.insert(ves2.end(), top_cell.cell_refs.at(cr).v_edges2.begin(),
                  top_cell.cell_refs.at(cr).v_edges2.end());
    }
    hidx1.emplace_back(hes1.size());
    vidx1.emplace_back(ves1.size());
    hidx2.emplace_back(hes2.size());
    vidx2.emplace_back(ves2.size());

    std::vector<event> events;
    events.reserve(rows[row].size() * 2);
    for (int i = 0; i < rows[row].size(); i++) {
      const auto& cell_ref = top_cell.cell_refs.at(rows[row][i]);
      const auto& the_cell = db.get_cell(cell_ref.cell_name);
      if (the_cell.is_touching(layers.back())) {  // layer 2 is via
        events.emplace_back(event{Intvl{cell_ref.mbr2[2], cell_ref.mbr2[3], i},
                                  cell_ref.mbr2[0], false, true});
        events.emplace_back(event{Intvl{cell_ref.mbr2[2], cell_ref.mbr2[3], i},
                                  cell_ref.mbr2[1], false, false});
      } else if (the_cell.is_touching(layers.front())) {  // layer 1 is metal
        events.emplace_back(event{Intvl{cell_ref.mbr1[2], cell_ref.mbr1[3], i},
                                  cell_ref.mbr1[0], true, true});
        events.emplace_back(event{Intvl{cell_ref.mbr1[2], cell_ref.mbr1[3], i},
                                  cell_ref.mbr1[1], true, false});
      } else {
        continue;
      }
    }

    {
      std::sort(events.begin(), events.end(),
                [](const auto& e1, const auto& e2) {
                  return e1.y == e2.y ? (e1.is_inevent and !e2.is_inevent)
                                      : e1.y < e2.y;
                });
    }

    core::interval_tree<int, int> tree_V;
    core::interval_tree<int, int> tree_M;
    tree_M.reverse = true;
    std::vector<std::pair<int, int>> ovlpairs;
    ovlpairs.reserve(events.size() * 2);

    for (int i = 0; i < events.size(); ++i) {
      const auto& e = events[i];
      if (e.is_polygon) {  // metal
        if (e.is_inevent) {
          tree_M.insert(e.intvl);
        } else {
          tree_M.remove(e.intvl);
        }
      } else {
        if (e.is_inevent) {
          tree_V.insert(e.intvl);
        } else {
          tree_V.remove(e.intvl);
          tree_M.get_intervals_overlapping_with(e.intvl, ovlpairs);
        }
      }
    }
    run_check_hedge(ovlpairs, hes1.data(), hes2.data(), hidx1.data(),
                    hidx2.data(), threshold, vios);
    run_check_vedge(ovlpairs, ves1.data(), ves2.data(), vidx1.data(),
                    vidx2.data(), threshold, vios);
  }
  // for (const auto& vio : vios) {
  //   std::cout << "e11x: " << vio.e11x << " e11y: " << vio.e11y
  //             << " e12x: " << vio.e12x << " e12y: " << vio.e12y
  //             << " e21x: " << vio.e21x << " e21y: " << vio.e21y
  //             << " e22x: " << vio.e22x << " e22y: " << vio.e22y << std::endl;
  // }
}

}  // namespace odrc