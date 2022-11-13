#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <iostream>
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

struct cpair {
  int c1;
  int c2;
};

__global__ void check_kernel(cpair*        cpairs,
                             int           size,
                             h_edge*       hes,
                             v_edge*       ves,
                             int*          hidx,
                             int*          vidx,
                             int           threshold,
                             check_result* result) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  int   c1         = cpairs[tid].c1;
  int   c2         = cpairs[tid].c2;
  int   h1s        = hidx[c1];
  int   h1e        = hidx[c1 + 1];
  int   h2s        = hidx[c2];
  int   h2e        = hidx[c2 + 1];
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
}

void run_check(cpair*        cpairs,
               int           size,
               h_edge*       hes,
               v_edge*       ves,
               int*          hidx,
               int*          vidx,
               int           threshold,
               check_result* result,
               cudaStream_t  stream) {
  int batch_size = 512;
  check_kernel<<<(size + batch_size - 1) / batch_size, batch_size, 0, stream>>>(
      cpairs, size, hes, ves, hidx, vidx, threshold, result);
}

void space_check_dac23(const odrc::core::database& db,
                       int                         layer1,
                       int                         layer2,
                       int                         threshold) {
  std::vector<h_edge> hes;
  std::vector<v_edge> ves;
  std::vector<int>    hidx;
  std::vector<int>    vidx;
  for (const auto& cr : db.cells.back().cell_refs) {
    hidx.emplace_back(hes.size());
    vidx.emplace_back(ves.size());
    hes.insert(hes.end(), cr.h_edges.begin(), cr.h_edges.end());
    ves.insert(ves.end(), cr.v_edges.begin(), cr.v_edges.end());
  }
  hidx.emplace_back(hes.size());
  vidx.emplace_back(ves.size());
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  h_edge*       h_edges;
  v_edge*       v_edges;
  int*          h_idx;
  int*          v_idx;
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
  cudaMallocAsync((void**)&results, sizeof(check_result) * 128000, stream1);

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
  }
  std::cout << "# events: " << events.size() << std::endl;

  {
    odrc::util::timer t("t", logger);
    t.start();

    std::sort(events.begin(), events.end(), [](const auto& e1, const auto& e2) {
      if (e1.y == e2.y) {
        return e1.is_inevent and !e2.is_inevent;
      } else {
        return e1.y < e2.y;
      }
    });
    t.pause();
  }

  core::interval_tree<int, int> tree;

  std::cout << "scanning bottom up" << std::endl;
  int                                   add = 0;
  int                                   del = 0;
  int                                   cnt = 0;
  int                                   sum = 0;
  std::unordered_map<int, const event*> map;
  int                                   total_ovlp = 0;
  {
    odrc::util::timer t("t", logger);
    t.start();
    long long sum = 0;
    for (auto& e : events) {
      sum += e.intvl.mid();
    }
    int mid = sum / double(events.size());
    std::cout << "MID: " << mid << std::endl;
    // hack to balance tree
    Intvl _i1{mid - 1, mid + 1, -1};
    Intvl _i2{mid * 0.5 - 1, mid * 0.5 + 1, -1};
    Intvl _i3{mid * 1.5 - 1, mid * 1.5 + 1, -1};
    Intvl _i4{mid * 0.25 - 1, mid * 0.25 + 1, -1};
    Intvl _i5{mid * 0.75 - 1, mid * 0.75 + 1, -1};
    Intvl _i6{mid * 1.25 - 1, mid * 1.25 + 1, -1};
    Intvl _i7{mid * 1.75 - 1, mid * 1.75 + 1, -1};
    tree.insert(_i1);
    tree.insert(_i2);
    tree.insert(_i3);
    tree.insert(_i4);
    tree.insert(_i5);
    tree.insert(_i6);
    tree.insert(_i7);
    tree.remove(_i1);
    tree.remove(_i2);
    tree.remove(_i3);
    tree.remove(_i4);
    tree.remove(_i5);
    tree.remove(_i6);
    tree.remove(_i7);
    int                next_progress = 5;
    std::vector<cpair> cpairs;
    int                batch_size = 10240;
    int                next_start = 0;
    int                next_send  = batch_size;
    for (int i = 0; i < events.size(); ++i) {
      if (double(i) / events.size() >= next_progress / 100.0) {
        std::cout << "Progress : " << next_progress << "%" << std::endl;
        next_progress += 5;
      }
      //   for (const auto& e : events) {
      const auto& e = events[i];
      if (e.is_inevent) {
        auto ovlp = tree.get_intervals_overlapping_with(e.intvl);
        for (int c : ovlp) {
          cpairs.emplace_back(cpair{c, e.intvl.v});
          if (cpairs.size() >= next_send) {
            cudaStreamSynchronize(stream1);
            run_check(&cpairs[next_start], batch_size, h_edges, v_edges, h_idx,
                      v_idx, threshold, results, stream1);
            // std::cout << "  async processing batch "
            //   << next_send / batch_size - 1 << std::endl;
            next_start = next_send;
            next_send += batch_size;
          }
        }
        tree.insert(e.intvl);
      } else {
        tree.remove(e.intvl);
      }
    }
    cudaStreamSynchronize(stream1);
    run_check(&cpairs[next_start], cpairs.size() - next_start, h_edges, v_edges,
              h_idx, v_idx, threshold, results, stream1);
  }
}
}  // namespace odrc