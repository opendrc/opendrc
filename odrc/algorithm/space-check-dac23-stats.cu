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
struct check_result {
  int  e11x;
  int  e11y;
  int  e12x;
  int  e12y;
  int  e21x;
  int  e21y;
  int  e22x;
  int  e22y;
  bool is_violation = false;
};

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
    if (parent[k] == k)
      return k;  // If i am my own parent/rep
    // Find with Path compression, meaning we update the parent for this node
    // once recursion returns
    parent[k] = Find(parent[k]);
    return parent[k];
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
};

struct cpair {
  int c1;
  int c2;
};

__global__ void brute_force_check(int           size,
                                  h_edge*       hes,
                                  v_edge*       ves,
                                  int*          hidx,
                                  int*          vidx,
                                  int*          mbrs,
                                  int           threshold,
                                  check_result* result) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  int l = mbrs[tid * 4];
  int r = mbrs[tid * 4 + 1];
  int b = mbrs[tid * 4 + 2];
  int u = mbrs[tid * 4 + 3];
  for (int i = 0; i < size; ++i) {
    int ll = mbrs[i * 4];
    int rr = mbrs[i * 4 + 1];
    int bb = mbrs[i * 4 + 2];
    int uu = mbrs[i * 4 + 3];
    if (rr < l or ll > r or uu < b or bb > u) {
      continue;
    }
    int   c1         = tid;
    int   c2         = i;
    int   h1s        = hidx[c1];
    int   h1e        = hidx[c1 + 1];
    int   h2s        = hidx[c2];
    int   h2e        = hidx[c2 + 1];
    int   v1s        = vidx[c1];
    int   v1e        = vidx[c1 + 1];
    int   v2s        = vidx[c2];
    int   v2e        = vidx[c2 + 1];
    int   res_offset = tid * 10;  // at most 10 violations per thread
    auto& res        = result[res_offset];
    for (int p1 = h1s; p1 < h1e; ++p1) {
      for (int p2 = h2s; p2 < h2e; ++p2) {
        // space
        res.e11x = hes[p1].x1;
        res.e11y = hes[p1].y;
        res.e12x = hes[p1].x2;
        res.e12y = hes[p1].y;
        res.e21x = hes[p2].x1;
        res.e21y = hes[p2].y;
        res.e22x = hes[p2].x2;
        res.e22y = hes[p2].y;

        if (res.e11y < res.e22y) {
          // e22 e21
          // e11 e12
          bool is_outside_to_outside =
              res.e11x < res.e12x and res.e21x > res.e22x;
          bool is_too_close = res.e21y - res.e11y < threshold;
          bool is_projection_overlap =
              res.e21x < res.e11x and res.e12x < res.e22x;
          res.is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e12 e11
          // e21 e22
          bool is_outside_to_outside =
              res.e21x < res.e22x and res.e11x > res.e12x;
          bool is_too_close = res.e11y - res.e21y < threshold;
          bool is_projection_overlap =
              res.e11x < res.e21x and res.e22x < res.e12x;
          res.is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        }
        if (res.is_violation) {
          ++res_offset;
          if (res_offset >= (tid + 1) * 10)
            return;
          res = result[res_offset];
        }
      }
    }
    for (int p1 = v1s; p1 < v1e; ++p1) {
      for (int p2 = v2s; p2 < v2e; ++p2) {
        res.e11x = ves[p1].x;
        res.e11y = ves[p1].y1;
        res.e12x = ves[p1].x;
        res.e12y = ves[p1].y2;
        res.e21x = ves[p2].x;
        res.e21y = ves[p2].y1;
        res.e22x = ves[p2].x;
        res.e22y = ves[p2].y2;
        if (res.e11x < res.e21x) {
          // e11 e22
          // e12 e21
          bool is_outside_to_outside =
              res.e11y > res.e12y and res.e21y < res.e22y;
          bool is_too_close = res.e21x - res.e11x < threshold;
          bool is_projection_overlap =
              res.e11y < res.e21y and res.e22y < res.e12y;
          res.is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
          if (res.is_violation) {
            printf("T[%d]: (%d, %d), (%d, %d), (%d, %d), (%d, %d)\n", tid,
                   res.e11x, res.e11y, res.e12x, res.e12y, res.e21x, res.e21y,
                   res.e22x, res.e22y);
          }

        } else {
          // e21 e12
          // e22 e11
          bool is_outside_to_outside =
              res.e21y > res.e22y and res.e11y < res.e21y;
          bool is_too_close = res.e11x - res.e21x < threshold;
          bool is_projection_overlap =
              res.e21y < res.e11y and res.e12y < res.e22y;
          res.is_violation =
              is_outside_to_outside and is_too_close and is_projection_overlap;
        }
        if (res.is_violation) {
          ++res_offset;
          if (res_offset >= (tid + 1) * 10)
            return;
          res = result[res_offset];
        }
      }
    }
  }
}

void space_check_dac23(const odrc::core::database& db,
                       int                         layer1,
                       int                         layer2,
                       int                         threshold) {
  cudaError_t         err;
  std::vector<h_edge> hes;
  std::vector<v_edge> ves;
  std::vector<int>    hidx;
  std::vector<int>    vidx;
  std::vector<int>    mbrs;
  for (const auto& cr : db.cells.back().cell_refs) {
    const auto& the_cell = db.get_cell(cr.cell_name);
    if (!the_cell.is_touching(layer1)) {
      continue;
    }
    hidx.emplace_back(hes.size());
    vidx.emplace_back(ves.size());
    hes.insert(hes.end(), cr.h_edges.begin(), cr.h_edges.end());
    ves.insert(ves.end(), cr.v_edges.begin(), cr.v_edges.end());
    for (int i = 0; i < 4; ++i) {
      mbrs.emplace_back(cr.mbr[i]);
    }
  }
  hidx.emplace_back(hes.size());
  vidx.emplace_back(ves.size());
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  h_edge*       h_edges;
  v_edge*       v_edges;
  int*          h_idx;
  int*          v_idx;
  int*          d_mbrs;
  check_result* results;
  cudaMallocAsync((void**)&h_edges, sizeof(h_edge) * hes.size(), stream1);
  cudaMallocAsync((void**)&v_edges, sizeof(v_edge) * ves.size(), stream1);
  cudaMemcpyAsync(h_edges, hes.data(), sizeof(h_edge) * hes.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(v_edges, ves.data(), sizeof(v_edge) * ves.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMallocAsync((void**)&h_idx, sizeof(int) * hidx.size(), stream1);
  cudaMallocAsync((void**)&v_idx, sizeof(int) * vidx.size(), stream1);
  cudaMemcpyAsync(h_idx, hidx.data(), sizeof(int) * hidx.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(v_idx, vidx.data(), sizeof(int) * vidx.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMallocAsync((void**)&results, sizeof(check_result) * hidx.size() * 10,
                  stream1);
  cudaMallocAsync((void**)&d_mbrs, sizeof(int) * mbrs.size(), stream1);
  cudaMemcpyAsync(d_mbrs, mbrs.data(), sizeof(int) * mbrs.size(),
                  cudaMemcpyHostToDevice, stream1);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "Setup error: " << cudaGetErrorString(err) << std::endl;
  }

  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  using Intvl = core::interval<int, int>;
  struct event {
    Intvl intvl;
    int   y;
    int   offset;
    bool  is_polygon;
    bool  is_inevent;
  };

  std::vector<event> events;

  const auto& top_cell = db.cells.back();

  std::vector<int>        l;
  std::vector<int>        r;
  std::vector<int>        d;
  std::vector<int>        u;
  std::vector<int>        w;
  std::vector<int>        h;
  std::vector<int>        cells;
  std::unordered_set<int> universe_setx;
  std::unordered_set<int> universe_sety;

  std::cout << "# cell_refs: " << top_cell.cell_refs.size() << std::endl;
  for (int i = 0; i < top_cell.cell_refs.size(); ++i) {
    const auto& cell_ref = top_cell.cell_refs.at(i);
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (!the_cell.is_touching(layer1) and !the_cell.is_touching(layer2)) {
      continue;
    }
    events.emplace_back(
        event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3],
                    i + int(db.cells.back().polygons.size()) * 0},
              cell_ref.mbr[0], i, false, true});
    events.emplace_back(
        event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3],
                    (i + int(db.cells.back().polygons.size()) * 0)},
              cell_ref.mbr[1], i, false, false});
    l.emplace_back(cell_ref.mbr[0]);
    r.emplace_back(cell_ref.mbr[1]);
    d.emplace_back(cell_ref.mbr[2]);
    u.emplace_back(cell_ref.mbr[3]);
    w.emplace_back(cell_ref.mbr[1] - cell_ref.mbr[0]);
    h.emplace_back(cell_ref.mbr[3] - cell_ref.mbr[2]);
    cells.emplace_back(i);
    universe_setx.insert(cell_ref.mbr[0]);
    universe_setx.insert(cell_ref.mbr[1]);
    universe_sety.insert(cell_ref.mbr[2]);
    universe_sety.insert(cell_ref.mbr[3]);
  }
  std::cout << l.size() << std::endl;
  std::cout << mbrs.size() << std::endl;
  std::cout << hidx.size() << std::endl;
  std::sort(w.begin(), w.end());
  std::sort(h.begin(), h.end());
  odrc::util::timer uft("uf", logger);
  uft.start();

  std::vector<int> universex(universe_setx.begin(), universe_setx.end());
  std::vector<int> universey(universe_sety.begin(), universe_sety.end());
  std::sort(universex.begin(), universex.end());
  std::sort(universey.begin(), universey.end());
  // for (int i = 0; i < universey.size() - 1; ++i) {
  //   std::cout << universey[i] << " " << universey[i + 1]
  //             << ", delta = " << universey[i + 1] - universey[i] <<
  //             std::endl;
  // }

  // x- UF
  std::unordered_map<int, int> xcomp;
  for (int i = 0; i < universex.size(); ++i) {
    xcomp[universex[i]] = i;
  }
  DisjointSet ufx;
  ufx.makeSet(universex.size());
  for (int i = 0; i < d.size(); ++i) {
    int ufl = xcomp.at(l[i]);
    int ufr = xcomp.at(r[i]);
    for (int k = ufl + 1; k <= ufr; ++k) {
      ufx.Union(ufl, k);
    }
  }

  // y-UF
  std::unordered_map<int, int> ycomp;
  for (int i = 0; i < universey.size(); ++i) {
    ycomp[universey[i]] = i;
  }
  DisjointSet ufy;
  ufy.makeSet(universey.size());
  for (int i = 0; i < d.size(); ++i) {
    int ufb = ycomp.at(d[i]);
    int ufu = ycomp.at(u[i]);
    for (int k = ufb + 1; k <= ufu; ++k) {
      ufy.Union(ufb, k);
    }
  }
  uft.pause();
  std::unordered_set<int>      dsx;
  std::unordered_map<int, int> dsy;
  for (auto& a : ufx.parent) {
    dsx.insert(a);
  }
  for (auto& a : ufy.parent) {
    dsy.emplace(a, dsy.size());
  }
  std::vector<std::vector<int>> rows(dsy.size());
  for (int i = 0; i < cells.size(); ++i) {
    int y_comped = ycomp.at(top_cell.cell_refs.at(cells.at(i)).mbr[2]);
    int p_idx    = dsy.at(ufy.Find(y_comped));
    rows.at(p_idx).emplace_back(i);
  }
  for (int i = 0; i < rows.size(); ++i) {
    std::cout << "row " << i << ": " << rows[i].size() << std::endl;
  }
  return;
  std::cout << "# cells: " << l.size() << std::endl;
  std::cout << "x universe size: " << universex.size() << std::endl;
  std::cout << "x disjoint sets: " << dsx.size() << std::endl;
  std::cout << "y universe size: " << universey.size() << std::endl;
  std::cout << "y disjoint sets: " << dsy.size() << std::endl;
  return;
  int magic_num = 0;
  std::cout << w.front() << " to " << *(w.end() - 1 - magic_num) << ", avg "
            << std::accumulate(w.begin(), w.end() - magic_num, 0) /
                   double(w.size() - magic_num)
            << std::endl;
  std::cout << h.front() << " to " << *(h.end() - 1 - magic_num) << ", avg "
            << std::accumulate(h.begin(), h.end() - magic_num, 0) /
                   double(h.size() - magic_num)
            << std::endl;
  std::cout << "##############\n";
  std::cout << "# num cells: " << db.cells.size() << std::endl;
  // for (int i = 0; i < h.size(); ++i) {
  //   if (i == 0 or h[i] != h[i - 1]) {
  //     std::cout << i << " " << h[i] << std::endl;
  //   }
  // }
  // for (int i = 0; i < w.size(); ++i) {
  //   if (i == 0 or w[i] != w[i - 1]) {
  //     std::cout << i << " " << w[i] << std::endl;
  //   }
  // }

  std::sort(l.begin(), l.end());
  std::sort(r.begin(), r.end());
  std::sort(d.begin(), d.end());
  std::sort(u.begin(), u.end());
  for (int i = 0; i < d.size(); ++i) {
    if (i == 0 or d[i] != d[i - 1]) {
      std::cout << i << " " << d[i] << std::endl;
    }
  }

  exit(0);

  double total_l = 0;
  double total_r = 0;
  double total_d = 0;
  double total_u = 0;
  double total_m = 0;
  double min_m   = 1;
  double max_m   = 0;
  for (int i = 0; i < top_cell.cell_refs.size(); ++i) {
    auto&       cell_ref = top_cell.cell_refs[i];
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (!the_cell.is_touching(layer1) and !the_cell.is_touching(layer2)) {
      continue;
    }
    // l_pos: remove all cells whos right endpoint < l
    int l_pos = std::distance(
        r.begin(), std::lower_bound(r.begin(), r.end(), cell_ref.mbr[0]));
    // r_pos: remove all cells whos left endpoint > r
    int r_pos = std::distance(
        std::upper_bound(l.begin(), l.end(), cell_ref.mbr[1]), l.end());
    int d_pos = std::distance(
        u.begin(), std::lower_bound(u.begin(), u.end(), cell_ref.mbr[2]));
    int u_pos = std::distance(
        std::upper_bound(d.begin(), d.end(), cell_ref.mbr[3]), d.end());

    double lp = double(l_pos) / double(r.size());
    double rp = double(r_pos) / double(l.size());
    double dp = double(d_pos) / double(u.size());
    double up = double(u_pos) / double(d.size());

    double m = lp;
    m        = std::max(m, rp);
    m        = std::max(m, dp);
    m        = std::max(m, up);
    total_l += lp;
    total_r += rp;
    total_d += dp;
    total_u += up;
    total_m += m;
    min_m = std::min(min_m, m);
    max_m = std::max(max_m, m);
    assert(m <= 1);
    assert(total_m / double(i + 1) <= 1);
  }
  std::cout << total_l / double(l.size()) << std::endl;
  std::cout << total_r / double(r.size()) << std::endl;
  std::cout << total_d / double(d.size()) << std::endl;
  std::cout << total_u / double(u.size()) << std::endl;
  std::cout << total_m / double(u.size()) << std::endl;
  std::cout << "minmax: " << min_m << " " << max_m << std::endl;
  exit(0);

  std::cout << "# events: " << events.size() << std::endl;

  {
    odrc::util::timer t("BF check", logger);
    t.start();

    // std::sort(events.begin(), events.end(), [](const auto& e1, const auto&
    // e2) {
    //   if (e1.y == e2.y) {
    //     return e1.is_inevent and !e2.is_inevent;
    //   } else {
    //     return e1.y < e2.y;
    //   }
    // });

    int bs = 128;
    brute_force_check<<<(l.size() + bs - 1) / bs, bs>>>(
        l.size(), h_edges, v_edges, h_idx, v_idx, d_mbrs, threshold, results);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << " CUDA bf error: " << cudaGetErrorString(err) << std::endl;
    }
    t.pause();
  }
}
}  // namespace odrc