#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <thrust/async/sort.h>
#include <thrust/device_reference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

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

__global__ void hcheck_kernel(h_edge*       h_edges,
                              int           his,
                              int           hie,
                              int           hjs,
                              int           hje,
                              int           threshold,
                              check_result* results,
                              int*          res_offset) {
  int tid     = threadIdx.x + blockIdx.x + blockDim.x;
  int total_i = hie - his;
  int total_j = hje - hjs;
  if (tid >= total_i * total_j)
    return;
  int  i            = tid / total_j;
  int  j            = tid % total_j;
  int  e11x         = h_edges[i].x1;
  int  e11y         = h_edges[i].y;
  int  e12x         = h_edges[i].x2;
  int  e12y         = h_edges[i].y;
  int  e21x         = h_edges[j].x1;
  int  e21y         = h_edges[j].y;
  int  e22x         = h_edges[j].x2;
  int  e22y         = h_edges[j].y;
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
  return;
  if (is_violation) {
    check_result& res = results[atomicAdd(res_offset, 1)];
    res.e11x          = e11x;
    res.e11y          = e11y;
    res.e12x          = e12x;
    res.e12y          = e12y;
    res.e21x          = e21x;
    res.e21y          = e21y;
    res.e22x          = e22x;
    res.e22y          = e22y;
    res.is_violation  = false;
  }
}

__global__ void vcheck_kernel(v_edge*       v_edges,
                              int           vis,
                              int           vie,
                              int           vjs,
                              int           vje,
                              int           threshold,
                              check_result* results,
                              int*          res_offset) {
  int tid     = threadIdx.x + blockIdx.x + blockDim.x;
  int total_i = vie - vis;
  int total_j = vje - vjs;
  if (tid >= total_i * total_j)
    return;
  int  i            = tid / total_j;
  int  j            = tid % total_j;
  int  e11x         = v_edges[i].x;
  int  e11y         = v_edges[i].y1;
  int  e12x         = v_edges[i].x;
  int  e12y         = v_edges[i].y2;
  int  e21x         = v_edges[j].x;
  int  e21y         = v_edges[j].y1;
  int  e22x         = v_edges[j].x;
  int  e22y         = v_edges[j].y2;
  bool is_violation = false;

  if (e11x < e21x) {
    // e11 e22
    // e12 e21
    bool is_outside_to_outside = e11y > e12y and e21y < e22y;
    bool is_too_close          = e21x - e11x < threshold;
    bool is_projection_overlap = e11y < e21y and e22y < e12y;
    is_violation =
        is_outside_to_outside and is_too_close and is_projection_overlap;
    if (is_violation) {
      printf("T[%d]: (%d, %d), (%d, %d), (%d, %d), (%d, %d)\n", tid, e11x, e11y,
             e12x, e12y, e21x, e21y, e22x, e22y);
    }

  } else {
    // e21 e12
    // e22 e11
    bool is_outside_to_outside = e21y > e22y and e11y < e21y;
    bool is_too_close          = e11x - e21x < threshold;
    bool is_projection_overlap = e21y < e11y and e12y < e22y;
    is_violation =
        is_outside_to_outside and is_too_close and is_projection_overlap;
  }

  if (is_violation) {
    check_result& res = results[atomicAdd(res_offset, 1)];
    res.e11x          = e11x;
    res.e11y          = e11y;
    res.e12x          = e12x;
    res.e12y          = e12y;
    res.e21x          = e21x;
    res.e21y          = e21y;
    res.e22x          = e22x;
    res.e22y          = e22y;
    res.is_violation  = false;
  }
}

__device__ inline void run_check(h_edge*       h_edges,
                                 v_edge*       v_edges,
                                 int*          h_idx,
                                 int*          v_idx,
                                 int*          mbrs_d,
                                 int           i,
                                 int           j,
                                 int           tid,
                                 int           threshold,
                                 int*          vio_offset,
                                 check_result* check_results,
                                 h_edge*       hbuf,
                                 v_edge*       vbuf) {
  int sum = 0;

  // if (i >= csize or j >= csize) {
  //   printf("thread %d check %d %d", tid, i, j);
  // }
  int jxmin = mbrs_d[j * 4];
  int jxmax = mbrs_d[j * 4 + 1];
  int jymin = mbrs_d[j * 4 + 2];
  int jymax = mbrs_d[j * 4 + 3];
  if (mbrs_d[j * 4] > mbrs_d[i * 4 + 1] or mbrs_d[j * 4 + 1] < mbrs_d[i * 4] or
      mbrs_d[j * 4 + 2] > mbrs_d[i * 4 + 3] or
      mbrs_d[j * 4 + 3] < mbrs_d[i * 4 + 2])
    return;

  int his = h_idx[i];
  int hie = h_idx[i + 1];
  int hjs = h_idx[j];
  int hje = h_idx[j + 1];
  int vis = v_idx[i];
  int vie = v_idx[i + 1];
  int vjs = v_idx[j];
  int vje = v_idx[j + 1];
  for (int i = 0; i < 50; ++i) {
    if (i >= hje)
      break;
    hbuf[i * blockDim.x + threadIdx.x] = h_edges[hjs + i];
    vbuf[i * blockDim.x + threadIdx.x] = v_edges[vjs + i];
  }
  // horizontal checks
  // hcheck_kernel<<<((hie - his) * (hje - hjs) + 127) / 128, 128>>>(
  //     h_edges, his, hie, hjs, hje, threshold, results, &my_res);
  // vcheck_kernel<<<((vie - vis) * (vje - vjs) + 127) / 128, 128>>>(
  //     v_edges, vis, vie, vjs, vje, threshold, results, &my_res);
  // h_edge hbufj[100];
  // v_edge vbufj[100];
  // memcpy(hbufj, h_edges + hjs, sizeof(h_edge) * (hje - hjs));
  // memcpy(vbufj, v_edges + vjs, sizeof(h_edge) * (vje - vjs));
  for (int i = hie; i < his; ++i) {
    int iy = h_edges[i].y;
    if (iy - threshold >= jymax)
      break;
    else if (iy + threshold <= jymin)
      continue;
    int e11x = h_edges[i].x1;
    int e11y = h_edges[i].y;
    int e12x = h_edges[i].x2;
    int e12y = h_edges[i].y;
    for (int j = 0; j < hje - hjs; ++j) {
      int jy = h_edges[j].y;
      if (jy - threshold >= iy)
        break;
      else if (jy + threshold <= iy)
        continue;
      int  e21x         = hbuf[j * blockDim.x + threadIdx.x].x1;
      int  e21y         = hbuf[j * blockDim.x + threadIdx.x].y;
      int  e22x         = hbuf[j * blockDim.x + threadIdx.x].x2;
      int  e22y         = hbuf[j * blockDim.x + threadIdx.x].y;
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
        int           offset = atomicAdd(vio_offset, 1);
        check_result& res    = check_results[offset];
        res.e11x             = e11x;
        res.e11y             = e11y;
        res.e12x             = e12x;
        res.e12y             = e12y;
        res.e21x             = e21x;
        res.e21y             = e21y;
        res.e22x             = e22x;
        res.e22y             = e22y;
        res.is_violation     = false;
      }
    }
  }
  for (int i = vie; i < vis; ++i) {
    int ix = v_edges[i].x;
    if (ix - threshold >= jxmax)
      break;
    else if (ix + threshold <= jxmin)
      continue;
    int e11x = v_edges[i].x;
    int e11y = v_edges[i].y1;
    int e12x = v_edges[i].x;
    int e12y = v_edges[i].y2;
    for (int j = 0; j < vje - vjs; ++j) {
      int jx = v_edges[j].x;
      if (jx - threshold >= ix)
        break;
      else if (jx + threshold <= ix)
        continue;
      int  e21x         = vbuf[j * blockDim.x + threadIdx.x].x;
      int  e21y         = vbuf[j * blockDim.x + threadIdx.x].y1;
      int  e22x         = vbuf[j * blockDim.x + threadIdx.x].x;
      int  e22y         = vbuf[j * blockDim.x + threadIdx.x].y2;
      bool is_violation = false;
      if (e11x < e21x) {
        // e11 e22
        // e12 e21
        bool is_outside_to_outside = e11y > e12y and e21y < e22y;
        bool is_too_close          = e21x - e11x < threshold;
        bool is_projection_overlap = e11y < e21y and e22y < e12y;
        is_violation =
            is_outside_to_outside and is_too_close and is_projection_overlap;
        if (is_violation) {
          printf("T[%d]: (%d, %d), (%d, %d), (%d, %d), (%d, %d)\n", tid, e11x,
                 e11y, e12x, e12y, e21x, e21y, e22x, e22y);
        }

      } else {
        // e21 e12
        // e22 e11
        bool is_outside_to_outside = e21y > e22y and e11y < e21y;
        bool is_too_close          = e11x - e21x < threshold;
        bool is_projection_overlap = e21y < e11y and e12y < e22y;
        is_violation =
            is_outside_to_outside and is_too_close and is_projection_overlap;
      }

      if (is_violation) {
        int           offset = atomicAdd(vio_offset, 1);
        check_result& res    = check_results[offset];
        res.e11x             = e11x;
        res.e11y             = e11y;
        res.e12x             = e12x;
        res.e12y             = e12y;
        res.e21x             = e21x;
        res.e21y             = e21y;
        res.e22x             = e22x;
        res.e22y             = e22y;
        res.is_violation     = false;
      }
    }
  }
}

struct event {
  int x;
  int y1;
  int y2;
  int id;
};

__global__ void run_row(h_edge*       h_edges,
                        v_edge*       v_edges,
                        int*          h_idx,
                        int*          v_idx,
                        int*          mbrs_d,
                        check_result* results,
                        event*        events,
                        int*          eidx,
                        int           nrows,
                        int           threshold,
                        int           start) {
  __shared__ int    vio_offset;
  __shared__ h_edge hbuf[32 * 50];
  __shared__ v_edge vbuf[32 * 50];
  int               tid = threadIdx.x + blockDim.x * blockIdx.x * 2 + start;
  if (tid >= nrows)
    return;
  if (tid == 0) {
    vio_offset = 0;
  }
  struct idv {
    int  id;
    bool is_valid = false;
  } idvs[10];
  int rs = eidx[tid];
  int re = eidx[tid + 1];
  __syncthreads();
  for (int i = rs; i < re; ++i) {
    event& e = events[i];
    if (e.id > 0) {  // check and insert
      int  eid  = e.id - 1;
      int* mbre = &mbrs_d[eid * 4];
      // self check
      run_check(h_edges, v_edges, h_idx, v_idx, mbrs_d, eid, eid, tid,
                threshold, &vio_offset, results, hbuf, vbuf);

      bool found = false;
      for (int j = 0; j < 10; ++j) {
        if (!idvs[j].is_valid) {  // place or skip
          if (found) {
            continue;
          }
          idvs[j].id       = eid;
          idvs[j].is_valid = true;
          found            = true;
        } else {
          run_check(h_edges, v_edges, h_idx, v_idx, mbrs_d, eid, idvs[j].id,
                    tid, threshold, &vio_offset, results, hbuf, vbuf);
        }
      }
    } else {
      for (int j = 0; j < 10; ++j) {
        if (idvs[j].id == -e.id - 1 and idvs[j].is_valid) {
          idvs[j].is_valid = false;
          break;
        }
      }
    }
  }
  // if (tid == 0) {
  //   __syncthreads();
  //   memcpy(results, shared_results, sizeof(check_result) * vio_offset);
  // }
}
__global__ void init(event* array, int* keys, int* indices, int size) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  keys[tid]    = array[tid].x;
  indices[tid] = tid;
}

void space_check_dac23(const odrc::core::database& db,
                       int                         layer1,
                       int                         layer2,
                       int                         threshold) {
  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  odrc::util::timer  t("uf", logger);
  odrc::util::timer  gt("gpu", logger);
  odrc::util::timer  tt("tree", logger);
  odrc::util::timer  loop("loop", logger);
  odrc::util::timer  h1("h1", logger);
  odrc::util::timer  h2("h2", logger);
  odrc::util::timer  h3("h3", logger);
  const auto&        cell_refs = db.cells.back().cell_refs;
  t.start();
  std::unordered_set<int> y;
  y.reserve(cell_refs.size() * 2);
  std::vector<int>    cells;
  std::vector<int>    lrs;
  std::vector<h_edge> hes;
  std::vector<v_edge> ves;
  std::vector<int>    hidx;
  std::vector<int>    vidx;
  std::vector<int>    mbrs;
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
    hidx.emplace_back(hes.size());
    vidx.emplace_back(ves.size());
    hes.insert(hes.end(), cr.h_edges.begin(), cr.h_edges.end());
    ves.insert(ves.end(), cr.v_edges.begin(), cr.v_edges.end());
    mbrs.emplace_back(cr.mbr[0]);
    mbrs.emplace_back(cr.mbr[1]);
    mbrs.emplace_back(cr.mbr[2]);
    mbrs.emplace_back(cr.mbr[3]);
  }
  hidx.emplace_back(hes.size());
  vidx.emplace_back(ves.size());
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  t.pause();

  std::cout << "loop through: " << t.get_elapsed() << std::endl;
  std::cout << "# cells: " << cells.size() << std::endl;
  std::cout << "hidx.size(): " << hidx.size() << std::endl;

  t.start();
  h_edge*       h_edges;
  v_edge*       v_edges;
  int*          h_idx;
  int*          v_idx;
  int*          cells_d;
  int*          mbrs_d;
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
  cudaMallocAsync((void**)&mbrs_d, sizeof(int) * mbrs.size(), stream1);
  cudaMemcpyAsync(mbrs_d, mbrs.data(), sizeof(int) * mbrs.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMallocAsync((void**)&cells_d, sizeof(int) * cells.size(), stream1);
  cudaMemcpyAsync(cells_d, cells.data(), sizeof(int) * cells.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMallocAsync((void**)&results, sizeof(check_result) * 150000, stream1);

  std::vector<int> yv(y.begin(), y.end());
  std::sort(yv.begin(), yv.end());
  std::vector<int> y_comp(yv.back() + 5);
  std::cout << yv.back() << std::endl;
  for (int i = 0; i < yv.size(); ++i) {
    y_comp[yv[i]] = i;
  }
  for (int i = 0; i < lrs.size(); ++i) {
    lrs[i] = y_comp[lrs[i]];
  }
  const int        csize = cells.size();
  std::vector<int> ufv(yv.size(), 0);
  std::iota(ufv.begin(), ufv.end(), 0);

  int lrs_size = lrs.size();
  for (int i = 0; i < lrs_size; i += 2) {
    int ufb  = lrs[i];
    int ufu  = lrs[i + 1];
    ufv[ufb] = ufv[ufb] > ufu ? ufv[ufb] : ufu;
  }
  int lidx = -1;

  int end = -1;
  for (int i = 0; i < ufv.size(); ++i) {
    if (i > end) {
      end = ufv[i];
      ++lidx;
    }
    end    = std::max(end, ufv[i]);
    ufv[i] = lidx;
  }
  std::vector<std::vector<int>> rows(lidx + 1);
  std::cout << "# rows: " << rows.size() << std::endl;
  loop.start();
  for (int i = 0; i < csize; ++i) {
    rows[ufv[lrs[i * 2]]].emplace_back(i);
  }
  cudaStreamSynchronize(stream1);
  t.pause();
  std::cout << "UF and data transfer: " << t.get_elapsed() << std::endl;
  loop.start();
  std::vector<event> events;
  std::vector<int>   eidx;
  events.reserve(cells.size() * 2);
  eidx.reserve(rows.size() + 1);
  eidx.emplace_back(0);
  event* events_d = nullptr;
  int*   eidx_d   = nullptr;
  cudaMallocAsync((void**)&events_d, cells.size() * sizeof(event) * 2, stream1);
  cudaMallocAsync((void**)&eidx_d, (rows.size() + 1) * sizeof(int), stream1);
  int bs = 128;
  for (int i = 0; i < rows.size(); ++i) {
    // if(i != 49) continue;
    int bs          = 128;
    int rsize       = rows[i].size();
    int total_check = rsize * (rsize + 1) / 2;
    h1.start();
    // int*   keys;
    // int*   indices;
    // event* tmp;
    // cudaMallocAsync((void**)&keys, sizeof(int) * rows[i].size() * 2,
    // stream1); cudaMallocAsync((void**)&indices, sizeof(int) * rows[i].size()
    // * 2,
    //                 stream1);
    // cudaMallocAsync((void**)&tmp, sizeof(event) * rows[i].size() * 2,
    // stream1);
    int start_offset = events.size();
    for (int j = 0; j < rows[i].size(); ++j) {
      int* mbrj = &mbrs[rows[i][j] * 4];
      events.emplace_back(event{mbrj[0], mbrj[2], mbrj[3], rows[i][j] + 1});
      events.emplace_back(event{mbrj[1], mbrj[2], mbrj[3], -rows[i][j] - 1});
    }
    int end_offset = events.size();
    eidx.emplace_back(end_offset);
    cudaStreamSynchronize(stream1);
    cudaMemcpyAsync(events_d + start_offset, &events[start_offset],
                    sizeof(event) * (end_offset - start_offset),
                    cudaMemcpyHostToDevice, stream1);
    h1.pause();
    h2.start();
    // event* array = thrust::raw_pointer_cast(events_d.data());
    // int    N     = events.size();
    // auto   serr  = cudaGetLastError();
    // init<<<(events.size() + bs - 1), bs, 0, stream1>>>(array, keys, indices,
    // N); cudaStreamSynchronize(stream1); thrust::sort_by_key(thrust::device,
    // keys, keys + 10, indices); thrust::copy_n(thrust::device,
    //                thrust::make_permutation_iterator(array, indices), N,
    //                tmp);
    // cudaStreamSynchronize(stream1);
    thrust::async::sort(thrust::device, events_d + start_offset,
                        events_d + end_offset,
                        [] __device__(const auto& e1, const auto& e2) {
                          return e1.x == e2.x ? e1.id > e2.id : e1.x < e2.x;
                        });
    h2.pause();
    // h3.start();
    // std::sort(events.begin(), events.end(), [](const auto& e1, const auto&
    // e2) {
    //   return e1.x == e2.x ? e1.id > e2.id : e1.x < e2.x;
    // });
    // h3.pause();
  }
  cudaMemcpy(eidx_d, eidx.data(), sizeof(int) * eidx.size(),
             cudaMemcpyHostToDevice);
  bs = 32;
  run_row<<<((rows.size() + 1) / 2 + bs - 1), bs>>>(
      h_edges, v_edges, h_idx, v_idx, mbrs_d, results, events_d, eidx_d,
      int(eidx.size()), threshold, 0);
  run_row<<<((rows.size() + 1) / 2 + bs - 1), bs>>>(
      h_edges, v_edges, h_idx, v_idx, mbrs_d, results, events_d, eidx_d,
      int(eidx.size()), threshold, 1);
  cudaDeviceSynchronize();
  loop.pause();
  return;
}
}  // namespace odrc