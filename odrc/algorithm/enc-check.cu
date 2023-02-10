#include <odrc/algorithm/parallel_mode.hpp>

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

#include <odrc/algorithm/layout-partition.hpp>
#include <odrc/core/interval_tree.hpp>

namespace odrc {

using edge     = odrc::core::edge;
using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;
using o_edge   = odrc::core::orthogonal_edge;

__global__ void hori_search(o_edge* hedges1,
                            o_edge* hedges2,
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
  int y   = hedges2[tid].intercept;
  int mid = 0;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_y = hedges1[mid].intercept;
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
    int mid_y = hedges1[mid].intercept;
    if (mid_y - y >= threshold) {  // too right
      r = mid - 1;
    } else {
      l = mid + 1;
    }
  }
  eends[tid] = mid;
}

__global__ void vert_search(o_edge* vedges1,
                            o_edge* vedges2,
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
  int x   = vedges2[tid].intercept;
  int mid = 0;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_x = vedges1[mid].intercept;
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
    int mid_x = vedges1[mid].intercept;
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
                           o_edge*       hedges1,
                           o_edge*       hedges2,
                           int           size,
                           int           threshold,
                           check_result* results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  int offset = tid * 2;
  // printf("hello");
  o_edge& e     = hedges2[tid];
  int     e11x  = e.p_start;
  int     e11y  = e.intercept;
  int     e12x  = e.p_end;
  int     e12y  = e.intercept;
  int     start = estarts[tid];
  int     end   = eends[tid];
  for (int i = start; i < end; ++i) {
    o_edge& ee           = hedges1[i];
    int     e21x         = ee.p_start;
    int     e21y         = ee.intercept;
    int     e22x         = ee.p_end;
    int     e22y         = ee.intercept;
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
}

__global__ void vert_check(int*          estarts,
                           int*          eends,
                           o_edge*       vedges1,
                           o_edge*       vedges2,
                           int           size,
                           int           threshold,
                           check_result* results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  int   offset = tid * 2;
  auto& e      = vedges2[tid];
  int   e11x   = e.intercept;
  int   e11y   = e.p_start;
  int   e12x   = e.intercept;
  int   e12y   = e.p_end;

  int start = estarts[tid];
  int end   = eends[tid];
  for (int i = start; i < end; ++i) {
    o_edge& ee           = vedges1[i];
    int     e21x         = ee.intercept;
    int     e21y         = ee.p_start;
    int     e22x         = ee.intercept;
    int     e22y         = ee.p_end;
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
}

void enc_check_par(odrc::core::database&         db,
                   int                           layer1,
                   int                           layer2,
                   int                           threshold,
                   std::vector<core::violation>& vios) {
  check_result* dresults;
  o_edge*       dh_edges1;
  o_edge*       dh_edges2;
  o_edge*       dv_edges1;
  o_edge*       dv_edges2;
  int*          dh_idx1;
  int*          dh_idx2;
  int*          dv_idx1;
  int*          dv_idx2;
  int*          hstart;
  int*          hend;
  int*          vstart;
  int*          vend;
  cudaStream_t  stream1;
  cudaStream_t  stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaMallocAsync((void**)&dresults, sizeof(check_result) * 100000, stream1);
  cudaMallocAsync((void**)&dh_edges1, sizeof(o_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dh_edges2, sizeof(o_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dv_edges1, sizeof(o_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dv_edges2, sizeof(o_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&dh_idx1, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&dh_idx2, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&dv_idx1, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&dv_idx2, sizeof(int) * 10000, stream1);
  cudaMallocAsync((void**)&hstart, sizeof(int) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&hend, sizeof(int) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&vstart, sizeof(int) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&vend, sizeof(int) * 5000 * 100, stream1);
  std::vector<check_result> res_host;

  auto rows = layout_partition(db, std::vector{layer1, layer2}, threshold);

  const auto& top_cell = db.get_top_cell();
  int         sum      = 0;
  for (int row = 0; row < rows.size(); row++) {
    std::vector<o_edge> hes1;
    std::vector<o_edge> hes2;
    std::vector<o_edge> ves1;
    std::vector<o_edge> ves2;
    std::vector<int>    hidx1;
    std::vector<int>    hidx2;
    std::vector<int>    vidx1;
    std::vector<int>    vidx2;

    for (int i = 0; i < rows[row].size(); i++) {
      int   cr       = rows[row][i];
      auto& cell_ref = top_cell.cell_refs.at(cr);
      auto& the_cell = db.get_cell(cell_ref.cell_name);
      hidx2.emplace_back(hes2.size());
      vidx2.emplace_back(ves2.size());
      if (the_cell.is_touching(layer1)) {
        hes2.insert(hes2.end(), cell_ref.left_edges.at(layer1).begin(),
                    cell_ref.left_edges.at(layer1).end());
        hes2.insert(hes2.end(), cell_ref.right_edges.at(layer1).begin(),
                    cell_ref.right_edges.at(layer1).end());
        ves2.insert(ves2.end(), cell_ref.lower_edges.at(layer1).begin(),
                    cell_ref.lower_edges.at(layer1).end());
        ves2.insert(ves2.end(), cell_ref.upper_edges.at(layer1).begin(),
                    cell_ref.upper_edges.at(layer1).end());
      }
    }
    hidx2.emplace_back(hes2.size());
    vidx2.emplace_back(ves2.size());
    std::sort(hes2.begin(), hes2.end(),
              [] __device__(const o_edge& h1, const o_edge& h2) {
                return h1.intercept < h2.intercept;
              });
    std::sort(ves2.begin(), ves2.end(),
              [] __device__(const o_edge& v1, const o_edge& v2) {
                return v1.intercept < v2.intercept;
              });

    for (int i = 0; i < rows[row].size(); i++) {
      int   cr       = rows[row][i];
      auto& cell_ref = top_cell.cell_refs.at(cr);
      auto& the_cell = db.get_cell(cell_ref.cell_name);
      hidx1.emplace_back(hes1.size());
      vidx1.emplace_back(ves1.size());
      if (the_cell.is_touching(layer2)) {
        hes1.insert(hes1.end(), cell_ref.left_edges.at(layer2).begin(),
                    cell_ref.left_edges.at(layer2).end());
        hes1.insert(hes1.end(), cell_ref.right_edges.at(layer2).begin(),
                    cell_ref.right_edges.at(layer2).end());
        ves1.insert(ves1.end(), cell_ref.lower_edges.at(layer2).begin(),
                    cell_ref.lower_edges.at(layer2).end());
        ves1.insert(ves1.end(), cell_ref.upper_edges.at(layer2).begin(),
                    cell_ref.upper_edges.at(layer2).end());
      }
    }
    hidx1.emplace_back(hes1.size());
    vidx1.emplace_back(ves1.size());
    std::sort(hes1.begin(), hes1.end(),
              [] __device__(const o_edge& h1, const o_edge& h2) {
                return h1.intercept < h2.intercept;
              });
    std::sort(ves1.begin(), ves1.end(),
              [] __device__(const o_edge& v1, const o_edge& v2) {
                return v1.intercept < v2.intercept;
              });
    if (hes2.size() < 1000) {
      for (int j = 0; j < ves2.size(); ++j) {
        auto [p_start, p_end, intercept] = ves2[j];
        auto k =
            std::lower_bound(ves1.begin(), ves1.end(), intercept - threshold,
                             [](const auto& v1, int intercept) {
                               return v1.intercept < intercept;
                             });
        auto kend =
            std::upper_bound(ves1.begin(), ves1.end(), intercept + threshold,
                             [](int intercept, const auto& v1) {
                               return intercept < v1.intercept;
                             });
        for (; k < kend; ++k) {
          o_edge& ee = *k;
          if (is_violation(ves2[j], ee, threshold)) {
            res_host.emplace_back(check_result{intercept, p_start, intercept,
                                               p_end, ee.intercept, ee.p_start,
                                               ee.intercept, ee.p_end, true});
          }
        }
      }

      for (int j = 0; j < hes2.size(); ++j) {
        auto [p_start, p_end, intercept] = ves2[j];
        auto k =
            std::lower_bound(hes1.begin(), hes1.end(), intercept - threshold,
                             [](const auto& v1, int intercept) {
                               return v1.intercept < intercept;
                             });
        auto kend =
            std::upper_bound(hes1.begin(), hes1.end(), intercept + threshold,
                             [](int intercept, const auto& v1) {
                               return intercept < v1.intercept;
                             });
        for (; k < kend; ++k) {
          o_edge& ee = *k;
          if (is_violation(hes2[j], ee, threshold)) {
            res_host.emplace_back(
                check_result{p_start, intercept, p_end, intercept, ee.p_start,
                             ee.intercept, ee.p_end, ee.intercept, true});
          }
        }
      }
      continue;
    } else {
      cudaMemcpyAsync(dh_edges2, hes2.data(), sizeof(o_edge) * hes2.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaMemcpyAsync(dv_edges2, ves2.data(), sizeof(o_edge) * ves2.size(),
                      cudaMemcpyHostToDevice, stream2);
      cudaStreamSynchronize(stream1);
      cudaMemcpyAsync(dh_idx2, hidx2.data(), sizeof(int) * hidx2.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaStreamSynchronize(stream2);
      cudaMemcpyAsync(dv_idx2, vidx2.data(), sizeof(int) * vidx2.size(),
                      cudaMemcpyHostToDevice, stream2);
      cudaStreamSynchronize(stream1);
      cudaMemcpyAsync(dh_edges1, hes1.data(), sizeof(o_edge) * hes1.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaStreamSynchronize(stream2);
      cudaMemcpyAsync(dv_edges1, ves1.data(), sizeof(o_edge) * ves1.size(),
                      cudaMemcpyHostToDevice, stream2);

      cudaStreamSynchronize(stream1);
      thrust::async::sort(thrust::device, dh_edges1, dh_edges1 + hes1.size(),
                          [] __device__(const o_edge& h1, const o_edge& h2) {
                            return h1.intercept < h2.intercept;
                          });
      cudaMemcpyAsync(dh_idx1, hidx1.data(), sizeof(int) * hidx1.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaStreamSynchronize(stream2);
      cudaMemcpyAsync(dv_idx1, vidx1.data(), sizeof(int) * vidx1.size(),
                      cudaMemcpyHostToDevice, stream2);
      thrust::async::sort(thrust::device, dv_edges1, dv_edges1 + ves1.size(),
                          [] __device__(const o_edge& v1, const o_edge& v2) {
                            return v1.intercept < v2.intercept;
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
    }
  }
  result_transform(vios, dresults, sizeof(dresults) / sizeof(check_result));
}

}  // namespace odrc