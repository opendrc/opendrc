#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <odrc/core/interval_tree.hpp>
#include <odrc/utility/logger.hpp>
#include <odrc/utility/timer.hpp>

namespace odrc {

using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;
using odrc::core::h_edge;
using odrc::core::v_edge;
void run_check_hedge(std::vector<std::pair<int, int>> ovlpairs,
                     h_edge*                          hes,
                     int*                             hidx,
                     int                              threshold) {
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
        auto e11x = hes[p1].x1;
        auto e11y = hes[p1].y;
        auto e12x = hes[p1].x2;
        auto e12y = hes[p1].y;
        auto e21x = hes[p2].x1;
        auto e21y = hes[p2].y;
        auto e22x = hes[p2].x2;
        auto e22y = hes[p2].y;
        if (e11y < e22y) {
          // e22 e21
          // e11 e12
          bool is_outside_to_outside = e11x < e12x and e22x > e21x;
          bool is_too_close          = e21y - e11y < threshold;
          bool is_projection_overlap = e22x < e11x and e12x < e21x;
          auto is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e12 e11
          // e21 e22
          bool is_outside_to_outside = e21x < e22x and e12x > e11x;
          bool is_too_close          = e11y - e21y < threshold;
          bool is_projection_overlap = e12x < e21x and e22x < e11x;
          bool is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        }
      }
    }
  }
}
void run_check_vedge(std::vector<std::pair<int, int>> ovlpairs,
                     v_edge*                          ves,
                     int*                             vidx,
                     int                              threshold) {
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
        auto e11y = ves[p1].y1;
        auto e11x = ves[p1].x;
        auto e12y = ves[p1].y2;
        auto e12x = ves[p1].x;
        auto e21y = ves[p2].y1;
        auto e21x = ves[p2].x;
        auto e22y = ves[p2].y2;
        auto e22x = ves[p2].x;
        if (e11x < e22x) {
          // e22 e21
          // e11 e12
          bool is_outside_to_outside = e11y < e12y and e22y > e21y;
          bool is_too_close          = e12x - e11x < threshold;
          bool is_projection_overlap = e22y < e11y and e12y < e21y;
          auto is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e12 e11
          // e21 e22
          bool is_outside_to_outside = e21y < e22y and e12y > e11y;
          bool is_too_close          = e11x - e21x < threshold;
          bool is_projection_overlap = e12y < e21y and e22y < e11y;
          bool is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        }
      }
    }
  }
}
using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;
using odrc::core::h_edge;
using odrc::core::v_edge;

class DisjointSet {
 public:
  std::vector<int> parent;
  std::vector<int> rank;

  // perform MakeSet operation
  void makeSet(int n) {
    parent.resize(n, 0);
    rank.resize(n, 0);
    for (int i = 0; i < n; ++i) {
      parent[i] = i;
    }
  }
  void makeSet(std::vector<int> const& universe) {
    assert(false);
    // create `n` disjoint sets (one for each item)
    for (int i : universe) {
      parent[i] = i;
      rank[i]   = 0;
    }
  }

  // Find the root of the set in which element `k` belongs
  int Find(int k) { return parent[k] == k ? k : parent[k] = Find(parent[k]); }

  // Perform Union of two subsets
  void Union(int u, int v) {
    // find the root of the sets in which elements `x` and `y` belongs
    int ru = Find(u);
    int rv = Find(v);
    if (ru == rv)
      return;
    if (rank[ru] > rank[rv]) {
      parent[rv] = parent[ru];
    } else if (rank[rv] > rank[ru]) {
      parent[ru] = parent[rv];
    } else {
      parent[rv] = parent[ru];
      rank[ru]++;
    }
  }
  void union_range(int start, int end) {
    int p = Find(start);
    for (int i = start; i <= end; ++i) {
      parent[i] = p;
    }
  }
};

void enclosing_check_dac23(const odrc::core::database& db,
                           int                         layer1,
                           int                         layer2,
                           int                         threshold) {
  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  odrc::util::timer  ti("Disjoint-set", logger);
  odrc::util::timer  tim("pre process", logger);
  odrc::util::timer  time("sum_time", logger);
  odrc::util::timer  timi0("sweepline", logger);
  odrc::util::timer  timi1("interval tree", logger);
  odrc::util::timer  timi2("space check", logger);

  ti.start();
  const auto& cell_refs = db.cells.back().cell_refs;

  std::unordered_set<int> y;
  y.reserve(cell_refs.size() * 2);

  std::vector<int> cells;
  std::vector<int> lrs;
  cells.reserve(cell_refs.size());
  lrs.reserve(cell_refs.size() * 2);
  int cnt         = 0;
  int polygon_cnt = 0;
  for (int i = 0; i < cell_refs.size(); ++i) {
    const auto& cr       = cell_refs[i];
    const auto& the_cell = db.get_cell(cr.cell_name);
    if (!the_cell.is_touching(layer1) and !the_cell.is_touching(layer2)) {
      continue;
    }
    cells.emplace_back(i);
    y.insert(cr.mbr[2]);
    y.insert(cr.mbr[3]);
    lrs.emplace_back(cr.mbr[2]);
    lrs.emplace_back(cr.mbr[3]);
  }

  std::vector<int> yv(y.begin(), y.end());
  std::sort(yv.begin(), yv.end());

  std::vector<int> y_comp(yv.back() + 5);
  // std::cout << yv.back() << std::endl;

  for (int i = 0; i < yv.size(); ++i) {
    // y_comp.emplace(yv[i], i);
    y_comp[yv[i]] = i;
  }
  for (int i = 0; i < lrs.size(); ++i) {
    lrs[i] = y_comp[lrs[i]];
  }

  const int        csize = cells.size();
  std::vector<int> ufv(y_comp.size(), 0);
  std::iota(ufv.begin(), ufv.end(), 0);

  // std::cout << "comp and sort: " << t.get_elapsed() << std::endl;

  int lrs_size = lrs.size();
  for (int i = 0; i < lrs_size; i += 2) {
    int ufb  = lrs[i];
    int ufu  = lrs[i + 1];
    ufv[ufb] = ufv[ufb] > ufu ? ufv[ufb] : ufu;
  }

  // std::cout << "uf: " << t.get_elapsed() << std::endl;
  int lidx  = -1;
  int label = 0;
  int start = 0;
  int end   = -1;
  for (int i = 0; i < ufv.size(); ++i) {
    if (i > end) {
      start = i;
      end   = ufv[i];
      ++lidx;
    }
    end    = std::max(end, ufv[i]);
    ufv[i] = lidx;
  }

  // std::cout << "update label: " << t.get_elapsed() << std::endl;

  std::vector<std::vector<int>> rows(yv.size());
  for (int i = 0; i < csize; ++i) {
    rows[ufv[lrs[i * 2]]].emplace_back(cells[i]);
  }
  // std::cout << rows.size() << std::endl;

  ti.pause();
  time.start();
  const auto& top_cell = db.cells.back();
  int         sum      = 0;

  for (int row = 0; row < rows.size(); row++) {
    std::vector<h_edge> hes;
    std::vector<v_edge> ves;
    std::vector<int>    hidx;
    std::vector<int>    vidx;
    for (int i = 0; i < rows[row].size(); i++) {
      int cr = rows[row][i];
      hidx.emplace_back(hes.size());
      vidx.emplace_back(ves.size());
      hes.insert(hes.end(), top_cell.cell_refs.at(cr).h_edges.begin(),
                 top_cell.cell_refs.at(cr).h_edges.end());
      ves.insert(ves.end(), top_cell.cell_refs.at(cr).v_edges.begin(),
                 top_cell.cell_refs.at(cr).v_edges.end());
    }
    hidx.emplace_back(hes.size());
    vidx.emplace_back(ves.size());

    using Intvl = core::interval<int, int>;
    struct event {
      Intvl intvl;
      int   y;
      bool  is_polygon;
      bool  is_inevent;
    };

    std::vector<event> events;
    events.reserve(rows[row].size() * 2);
    timi0.start();
    for (int i = 0; i < rows[row].size(); i++) {
      const auto& cell_ref = top_cell.cell_refs.at(rows[row][i]);
      const auto& the_cell = db.get_cell(cell_ref.cell_name);
      if (the_cell.is_touching(layer1)) {
        events.emplace_back(event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3], i},
                                  cell_ref.mbr[0], false, true});
        events.emplace_back(event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3], i},
                                  cell_ref.mbr[1], false, false});
      } else if (the_cell.is_touching(layer2)) {
        events.emplace_back(event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3], i},
                                  cell_ref.mbr[0], true, true});
        events.emplace_back(event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3], i},
                                  cell_ref.mbr[1], true, false});
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
    timi0.pause();

    timi1.start();
    core::interval_tree<int, int>    tree_V;
    core::interval_tree<int, int>    tree_M;
    std::vector<std::pair<int, int>> ovlpairs;
    ovlpairs.reserve(events.size() * 2);

    for (int i = 0; i < events.size(); ++i) {
      const auto& e = events[i];
      if (e.is_polygon) {
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
    sum += ovlpairs.size();
    timi1.pause();
    timi2.start();
    run_check_hedge(ovlpairs, hes.data(), hidx.data(), threshold);
    run_check_vedge(ovlpairs, ves.data(), vidx.data(), threshold);
    timi2.pause();
  }
  time.pause();
}

}  // namespace odrc