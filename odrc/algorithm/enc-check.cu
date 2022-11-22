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

__global__ void hori_search(h_edge* hedges1,
                            h_edge* hedges2,
                            int     size1,
                            int     size2,
                            int     threshold,
                            int*    estarts,
                            int*    eends) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size2)
    return;
  int l   = 0;
  int r   = size1 - 1;
  int y   = hedges2[tid].y;
  int mid = 0;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_y = hedges1[mid].y;
    if (y - mid_y >= threshold) {  // too left
      l = mid + 1;
    } else {
      r = mid - 1;
    }
  }
  estarts[tid] = mid;
  l            = mid;
  r            = size1 - 1;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_y = hedges1[mid].y;
    if (mid_y - y >= threshold) {  // too right
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  eends[tid] = mid;
}

__global__ void vert_search(v_edge* vedges1,
                            v_edge* vedges2,
                            int     size1,
                            int     size2,
                            int     threshold,
                            int*    estarts,
                            int*    eends) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size2)
    return;
  int l   = 0;
  int r   = size1 - 1;
  int x   = vedges2[tid].x;
  int mid = 0;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_x = vedges1[mid].x;
    if (x - mid_x >= threshold) {
      l = mid + 1;
    } else {
      r = mid - 1;
    }
  }
  estarts[tid] = mid;
  l            = mid;
  r            = size1 - 1;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_x = vedges1[mid].x;
    if (mid_x - x >= threshold) {
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  eends[tid] = mid;
}

__global__ void hori_check(int*          estarts,
                           int*          eends,
                           h_edge*       hedges1,
                           h_edge*       hedges2,
                           int           size,
                           int           threshold,
                           check_result* results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  // __shared__ check_result shared_result[128 * 10];
  // for (int i = 0; i < 10; ++i) {
  //   shared_result[tid + blockDim.x * i].is_violation = false;
  // }
  int offset = tid * 2;
  // printf("hello");
  h_edge& e     = hedges2[tid];
  int     e11x  = e.x1;
  int     e11y  = e.y;
  int     e12x  = e.x2;
  int     e12y  = e.y;
  int     start = estarts[tid];
  int     end   = eends[tid];
  for (int i = start; i < end; ++i) {
    h_edge& ee           = hedges1[i];
    int     e21x         = ee.x1;
    int     e21y         = ee.y;
    int     e22x         = ee.x2;
    int     e22y         = ee.y;
    bool    is_violation = false;
    // if (e11y - e21y >= threshold)
    //   break;

    if (e11y < e22y) {
      // e22 e21
      // e12 e11
      bool is_inside_to_outside  = e11x > e12x and e21x > e22x;
      bool is_too_close          = e21y - e11y < threshold;
      bool is_projection_overlap = e22x < e11x and e12x < e21x;
      is_violation =
          is_inside_to_outside and is_too_close and is_projection_overlap;
    } else {
      // e11 e12
      // e21 e22
      bool is_inside_to_outside  = e11x < e12x and e21x < e22x;
      bool is_too_close          = e11y - e21y < threshold;
      bool is_projection_overlap = e11x < e22x and e21x < e12x;
      is_violation =
          is_inside_to_outside and is_too_close and is_projection_overlap;
    }

    if (is_violation) {
      if (offset < tid * 2 + 2 and offset < 100000) {
        // check_result& res = results[offset];  // shared_result[offset];
        check_result res;
        res.e11x         = e11x;
        res.e11y         = e11y;
        res.e12x         = e12x;
        res.e12y         = e12y;
        res.e21x         = e21x;
        res.e21y         = e21y;
        res.e22x         = e22x;
        res.e22y         = e22y;
        res.is_violation = true;
        ++offset;
        memcpy(results + offset, &res, sizeof(check_result));
      }
    }
  }
  // if (tid == 0) {
  //   __syncthreads();
  //   memcpy(results, shared_result, sizeof(check_result) * 128);
  // }
}

__global__ void vert_check(int*          estarts,
                           int*          eends,
                           v_edge*       vedges1,
                           v_edge*       vedges2,
                           int           size,
                           int           threshold,
                           check_result* results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  // __shared__ check_result shared_result[128 * 10];
  // for (int i = 0; i < 10; ++i) {
  //   shared_result[tid + blockDim.x * i].is_violation = false;
  // }
  int   offset = tid * 2;
  auto& e      = vedges2[tid];
  int   e11x   = e.x;
  int   e11y   = e.y1;
  int   e12x   = e.x;
  int   e12y   = e.y2;

  int start = estarts[tid];
  int end   = eends[tid];
  for (int i = start; i < end; ++i) {
    v_edge& ee           = vedges1[i];
    int     e21x         = ee.x;
    int     e21y         = ee.y1;
    int     e22x         = ee.x;
    int     e22y         = ee.y2;
    bool    is_violation = false;

    if (e11x < e22x) {
      // e12 e22
      // e11 e21
      bool is_inside_to_outside  = e11y < e12y and e22y > e21y;
      bool is_too_close          = e12x - e11x < threshold;
      bool is_projection_overlap = e22y > e11y and e12y > e21y;
      is_violation =
          is_inside_to_outside and is_too_close and is_projection_overlap;
    } else {
      // e21 e11
      // e22 e12
      bool is_inside_to_outside  = e21y > e22y and e11y > e11y;
      bool is_too_close          = e11x - e21x < threshold;
      bool is_projection_overlap = e12y < e21y and e22y < e11y;
      is_violation =
          is_inside_to_outside and is_too_close and is_projection_overlap;
    }

    if (is_violation) {
      if (offset < tid * 2 + 2 and offset < 100000) {
        check_result& res = results[offset];
        res.e11x          = e11x;
        res.e11y          = e11y;
        res.e12x          = e12x;
        res.e12y          = e12y;
        res.e21x          = e21x;
        res.e21y          = e21y;
        res.e22x          = e22x;
        res.e22y          = e22y;
        res.is_violation  = true;
        ++offset;
      }
    }
  }
  // if (tid == 0) {
  //   __syncthreads();
  //   memcpy(results + 100000, shared_result, sizeof(check_result) *
  //   blockDim.x);
  // }
}

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
          bool is_projection_overlap = e22x < e11x and e12x < e21x;
          is_violation =
              is_inside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e11 e12
          // e21 e22
          bool is_inside_to_outside  = e11x < e12x and e21x < e22x;
          bool is_too_close          = e11y - e21y < threshold;
          bool is_projection_overlap = e11x < e22x and e21x < e12x;
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
          bool is_projection_overlap = e22y > e11y and e12y > e21y;
          is_violation =
              is_inside_to_outside and is_too_close and is_projection_overlap;
        } else {
          // e21 e11
          // e22 e12
          bool is_inside_to_outside  = e21y > e22y and e11y > e12y;
          bool is_too_close          = e11x - e21x < threshold;
          bool is_projection_overlap = e12y < e21y and e22y < e11y;
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
using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;
using odrc::core::h_edge;
using odrc::core::v_edge;

void enclosing_check_dac23(const odrc::core::database& db,
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
  odrc::util::timer  h4("h4", logger);
  check_result*      dresults;
  h_edge*            dh_edges1;
  h_edge*            dh_edges2;
  v_edge*            dv_edges1;
  v_edge*            dv_edges2;
  int*               dh_idx1;
  int*               dh_idx2;
  int*               dv_idx1;
  int*               dv_idx2;
  int*               hstart;
  int*               hend;
  int*               vstart;
  int*               vend;
  cudaError_t        err;
  cudaStream_t       stream1;
  cudaStream_t       stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaMallocAsync((void**)&dresults, sizeof(check_result) * 100000, stream1);
  cudaMallocAsync((void**)&dh_edges1, sizeof(h_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dh_edges2, sizeof(h_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dv_edges1, sizeof(v_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dv_edges2, sizeof(v_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dh_idx1, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&dh_idx2, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&dv_idx1, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&dv_idx2, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&hstart, sizeof(int) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&hend, sizeof(int) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&vstart, sizeof(int) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&vend, sizeof(int) * 5000 * 100, stream1);
  std::vector<check_result> res_host;
  h1.start();

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
  std::cout << rows.size() << std::endl;

  const auto& top_cell = db.cells.back();
  int         sum      = 0;

  loop.start();
  for (int row = 0; row < rows.size(); row++) {
    std::vector<h_edge> hes1;
    std::vector<h_edge> hes2;
    std::vector<v_edge> ves1;
    std::vector<v_edge> ves2;
    std::vector<int>    hidx1;
    std::vector<int>    hidx2;
    std::vector<int>    vidx1;
    std::vector<int>    vidx2;
    h4.start();
    for (int i = 0; i < rows[row].size(); i++) {
      int cr = rows[row][i];
      hidx2.emplace_back(hes2.size());
      vidx2.emplace_back(ves2.size());
      hes2.insert(hes2.end(), top_cell.cell_refs.at(cr).h_edges2.begin(),
                  top_cell.cell_refs.at(cr).h_edges2.end());
      ves2.insert(ves2.end(), top_cell.cell_refs.at(cr).v_edges2.begin(),
                  top_cell.cell_refs.at(cr).v_edges2.end());
    }
    hidx2.emplace_back(hes2.size());
    vidx2.emplace_back(ves2.size());
    if (hes2.size() > 1000) {
      std::sort(hes2.begin(), hes2.end(),
                [] __device__(const h_edge& h1, const h_edge& h2) {
                  return h1.y < h2.y;
                });
      std::sort(ves2.begin(), ves2.end(),
                [] __device__(const v_edge& v1, const v_edge& v2) {
                  return v1.x < v2.x;
                });
      cudaMemcpyAsync(dh_edges2, hes2.data(), sizeof(h_edge) * hes2.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaMemcpyAsync(dv_edges2, ves2.data(), sizeof(v_edge) * ves2.size(),
                      cudaMemcpyHostToDevice, stream2);
      cudaMemcpyAsync(dh_idx2, hidx2.data(), sizeof(int) * hidx2.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaMemcpyAsync(dv_idx2, vidx2.data(), sizeof(int) * vidx2.size(),
                      cudaMemcpyHostToDevice, stream2);
    }
    for (int i = 0; i < rows[row].size(); i++) {
      int cr = rows[row][i];
      hidx1.emplace_back(hes1.size());
      vidx1.emplace_back(ves1.size());
      hes1.insert(hes1.end(), top_cell.cell_refs.at(cr).h_edges1.begin(),
                  top_cell.cell_refs.at(cr).h_edges1.end());
      ves1.insert(ves1.end(), top_cell.cell_refs.at(cr).v_edges1.begin(),
                  top_cell.cell_refs.at(cr).v_edges1.end());
    }
    hidx1.emplace_back(hes1.size());
    vidx1.emplace_back(ves1.size());
    h4.pause();
    err = cudaGetLastError();
    // std::cout << cudaGetErrorString(err) << std::endl;

    // std::cout << "loop through: " << t.get_elapsed() << std::endl;
    // std::cout << "# cells: " << cells.size() << std::endl;
    // std::cout << "hidx.size(): " << hidx.size() << std::endl;
    if (hes2.size() < 1000) {
      h3.start();
      std::sort(hes1.begin(), hes1.end(),
                [] __device__(const h_edge& h1, const h_edge& h2) {
                  return h1.y < h2.y;
                });
      std::sort(hes2.begin(), hes2.end(),
                [] __device__(const h_edge& h1, const h_edge& h2) {
                  return h1.y < h2.y;
                });
      std::sort(ves1.begin(), ves1.end(),
                [] __device__(const v_edge& v1, const v_edge& v2) {
                  return v1.x < v2.x;
                });
      std::sort(ves2.begin(), ves2.end(),
                [] __device__(const v_edge& v1, const v_edge& v2) {
                  return v1.x < v2.x;
                });
      for (int j = 0; j < ves2.size(); ++j) {
        auto& e    = ves2[j];
        int   e11x = e.x;
        int   e11y = e.y1;
        int   e12x = e.x;
        int   e12y = e.y2;
        auto  k =
            std::lower_bound(ves1.begin(), ves1.end(), e11x - threshold,
                             [](const auto& v1, int x) { return v1.x < x; });
        auto kend =
            std::upper_bound(ves1.begin(), ves1.end(), e11x + threshold,
                             [](int x, const auto& v1) { return x < v1.x; });
        for (; k < kend; ++k) {
          if (e11x - k->x >= threshold)
            continue;
          if (k->x - e11x >= threshold)
            break;
          v_edge& ee           = *k;
          int     e21x         = ee.x;
          int     e21y         = ee.y1;
          int     e22x         = ee.x;
          int     e22y         = ee.y2;
          bool    is_violation = false;

          if (e11x < e22x) {
            // e12 e22
            // e11 e21
            bool is_inside_to_outside  = e11y < e12y and e22y > e21y;
            bool is_too_close          = e12x - e11x < threshold;
            bool is_projection_overlap = e22y > e11y and e12y > e21y;
            is_violation =
                is_inside_to_outside and is_too_close and is_projection_overlap;
          } else {
            // e21 e11
            // e22 e12
            bool is_inside_to_outside  = e21y > e22y and e11y > e12y;
            bool is_too_close          = e11x - e21x < threshold;
            bool is_projection_overlap = e12y < e21y and e22y < e11y;
            is_violation =
                is_inside_to_outside and is_too_close and is_projection_overlap;
          }

          if (is_violation) {
            res_host.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x,
                                               e21y, e22x, e22y, true});
          }
        }
      }

      for (int j = 0; j < hes2.size(); ++j) {
        auto& e    = hes2[j];
        int   e11x = e.x1;
        int   e11y = e.y;
        int   e12x = e.x2;
        int   e12y = e.y;
        auto  k =
            std::lower_bound(hes1.begin(), hes1.end(), e11x - threshold,
                             [](const auto& h1, int y) { return h1.y < y; });
        auto kend =
            std::upper_bound(hes1.begin(), hes1.end(), e11x + threshold,
                             [](int y, const auto& h1) { return y < h1.y; });
        for (; k < kend; ++k) {
          if (e11y - k->y >= threshold)
            continue;
          if (k->y - e11y >= threshold)
            break;
          h_edge& ee           = *k;
          int     e21x         = ee.x1;
          int     e21y         = ee.y;
          int     e22x         = ee.x2;
          int     e22y         = ee.y;
          bool    is_violation = false;

          if (e11y < e22y) {
            // e22 e21
            // e12 e11
            bool is_inside_to_outside  = e11x > e12x and e21x > e22x;
            bool is_too_close          = e21y - e11y < threshold;
            bool is_projection_overlap = e22x < e11x and e12x < e21x;
            is_violation =
                is_inside_to_outside and is_too_close and is_projection_overlap;
          } else {
            // e11 e12
            // e21 e22
            bool is_inside_to_outside  = e11x < e12x and e21x < e22x;
            bool is_too_close          = e11y - e21y < threshold;
            bool is_projection_overlap = e11x < e22x and e21x < e12x;
            is_violation =
                is_inside_to_outside and is_too_close and is_projection_overlap;
          }

          if (is_violation) {
            res_host.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x,
                                               e21y, e22x, e22y, true});
          }
        }
      }

      h3.pause();
      continue;
    } else {
      h2.start();
      // cudaStreamSynchronize(stream1);
      // cudaStreamSynchronize(stream2);
      cudaMemcpyAsync(dh_edges1, hes1.data(), sizeof(h_edge) * hes1.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaMemcpyAsync(dv_edges1, ves1.data(), sizeof(v_edge) * ves1.size(),
                      cudaMemcpyHostToDevice, stream2);
      // thrust::async::sort(thrust::device, dh_edges2, dh_edges2 + hes2.size(),
      //                     [] __device__(const h_edge& h1, const h_edge& h2) {
      //                       return h1.y < h2.y;
      //                     });
      // thrust::async::sort(thrust::device, dv_edges2, dv_edges2 + ves2.size(),
      //                     [] __device__(const v_edge& v1, const v_edge& v2) {
      //                       return v1.x < v2.x;
      //                     });
      cudaStreamSynchronize(stream1);
      thrust::async::sort(thrust::device, dh_edges1, dh_edges1 + hes1.size(),
                          [] __device__(const h_edge& h1, const h_edge& h2) {
                            return h1.y < h2.y;
                          });
      cudaMemcpyAsync(dh_idx1, hidx1.data(), sizeof(int) * hidx1.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaStreamSynchronize(stream2);
      cudaMemcpyAsync(dv_idx1, vidx1.data(), sizeof(int) * vidx1.size(),
                      cudaMemcpyHostToDevice, stream2);
      thrust::async::sort(thrust::device, dv_edges1, dv_edges1 + ves1.size(),
                          [] __device__(const v_edge& v1, const v_edge& v2) {
                            return v1.x < v2.x;
                          });
      cudaDeviceSynchronize();
      int bs = 512;
      hori_search<<<(hes2.size() + bs - 1) / bs, bs, 0, stream1>>>(
          dh_edges1, dh_edges2, hes1.size(), hes2.size(), threshold, hstart,
          hend);
      vert_search<<<(ves2.size() + bs - 1) / bs, bs, 0, stream2>>>(
          dv_edges1, dv_edges2, ves1.size(), ves2.size(), threshold, vstart,
          vend);
      hori_check<<<(hes2.size() + bs - 1) / bs, bs, 0, stream1>>>(
          hstart, hend, dh_edges1, dh_edges2, hes2.size(), threshold, dresults);
      vert_check<<<(ves2.size() + bs - 1) / bs, bs, 0, stream2>>>(
          vstart, vend, dv_edges1, dv_edges2, ves2.size(), threshold, dresults);
      h2.pause();
    }
  }
  loop.pause();
  h1.pause();
}

}  // namespace odrc