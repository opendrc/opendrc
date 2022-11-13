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
  int Find(int k) {
    return parent[k] == k ? k : parent[k] = Find(parent[k]);
    // if (parent[k] == k)
    //   return k;  // If i am my own parent/rep
    // Find with Path compression, meaning we update the parent for this node
    // once recursion returns
    // parent[k] = Find(parent[k]);
    // return parent[k];
  }

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

void space_check_dac23(const odrc::core::database& db,
                       int                         layer1,
                       int                         layer2,
                       int                         threshold) {
  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  odrc::util::timer  t("uf", logger);
  odrc::util::timer  lu("lu", logger);
  odrc::util::timer  is("is", logger);
  odrc::util::timer  loop("loop", logger);
  const auto&        cell_refs = db.cells.back().cell_refs;
  t.start();
  std::unordered_set<int> y;
  y.reserve(cell_refs.size() * 2);
  std::vector<int> cells;
  //   std::unordered_map<int, int> y_comp;
  std::vector<int> lrs;
  cells.reserve(cell_refs.size());
  lrs.reserve(cell_refs.size() * 2);
  for (int i = 0; i < cell_refs.size(); ++i) {
    const auto& cr       = cell_refs[i];
    const auto& the_cell = db.get_cell(cr.cell_name);
    if (!the_cell.is_touching(layer1)) {
      continue;
    }
    cells.emplace_back(i);
    y.insert(cr.mbr[2]);
    y.insert(cr.mbr[3]);
    lrs.emplace_back(cr.mbr[2]);
    lrs.emplace_back(cr.mbr[3]);
  }
  t.pause();
  std::cout << "loop through: " << t.get_elapsed() << std::endl;
  t.start();
  std::vector<int> yv(y.begin(), y.end());
  std::sort(yv.begin(), yv.end());
  std::vector<int> y_comp(yv.back()+5);
  std::cout << yv.back() << std::endl;
  return;
  for (int i = 0; i < yv.size(); ++i) {
    // y_comp.emplace(yv[i], i);
    y_comp[yv[i]] = i;
  }
  for (int i = 0; i < lrs.size(); ++i) {
    lrs[i] = y_comp[lrs[i]];
  }
  //   DisjointSet uf;
  //   uf.makeSet(yv.size());
  // hack
  /*

  std::vector<int> ufv(10, 0);
  std::iota(ufv.begin(), ufv.end(), 0);
  int a[3][2] = {{0, 4}, {5, 9}, {4, 4}};
  for(int i = 0; i < 3; ++i) {
    int l = a[i][0];
    int r = a[i][1];
    ufv[l] = std::max(ufv[l], r);
  }
  for(int i = 0; i < 10; ++i) {
    std::cout << ufv[i] << " ";
  }
  std::cout << std::endl;
  int my_idx  = -1;
  int my_end   = -1;
  for (int i = 0; i < ufv.size(); ++i) {
    if (i > my_end) {
      my_end   = ufv[i];
      ++my_idx;
    }
    my_end    = std::max(my_end, ufv[i]);
    ufv[i] = my_idx;
  }
  for(int i = 0; i < 10; ++i) {
    std::cout << ufv[i] << " ";
  }
  std::cout << std::endl;
  return;
  */

  const int        csize = cells.size();
  std::vector<int> ufv(y_comp.size(), 0);
  std::iota(ufv.begin(), ufv.end(), 0);

  t.pause();
  std::cout << "comp and sort: " << t.get_elapsed() << std::endl;
  t.start();
  int lrs_size = lrs.size();
  for (int i = 0; i < lrs_size; i += 2) {
    // int ufb  = y_comp[cell_refs[cells[i]].mbr[2]];
    // int ufu  = y_comp[cell_refs[cells[i]].mbr[3]];
    int ufb  = lrs[i];
    int ufu  = lrs[i + 1];
    ufv[ufb] = ufv[ufb] > ufu ? ufv[ufb] : ufu;
    // for (int k = ufb + 1; k <= ufu; ++k) {
    //   uf.Union(ufb, k);
    // }
    // uf.union_range(ufb, ufu);
  }
  t.pause();
  std::cout << "uf: " << t.get_elapsed() << std::endl;
  t.start();
  int lidx  = -1;
  int label = 0;
  /*
  for (int i = 0; i < uf.parent.size(); ++i) {
    int p = uf.Find(i);
    if (i == 0 or p != label) {
      label = p;
      ++lidx;
    }
    uf.parent[i] = lidx;
  }
  */
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
  t.pause();
  std::cout << "update label: " << t.get_elapsed() << std::endl;
  t.start();
  // UF.find is invalid from this point
  std::vector<std::vector<int>> rows(yv.size());
  loop.start();
  for (int i = 0; i < csize; ++i) {
    // lu.start();
    // int y_comped = y_comp.at(cell_refs.at(cells.at(i)).mbr[2]);
    // is.start();
    // rows.at(uf.parent.at(y_comped)).emplace_back(i);
    rows[ufv[lrs[i * 2]]].emplace_back(cells[i]);
    // is.pause();
    // lu.pause();
  }
  loop.pause();
  t.pause();
  std::cout << "all: " << t.get_elapsed() << std::endl;
}

}  // namespace odrc