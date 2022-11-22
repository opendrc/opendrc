
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
}

void enclosing_check_dac23(const odrc::core::database& db,
                           int                         layer1,
                           int                         layer2,
                           int                         threshold) {
  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  odrc::util::timer  t("xcheck", logger);
  t.start();
  cudaError_t   err;
  check_result* dres;
  cudaStream_t  stream1 = nullptr;
  cudaStream_t  stream2 = nullptr;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  h_edge* dhes1;
  v_edge* dves1;
  h_edge* dhes2;
  v_edge* dves2;
  int*    hstart;
  int*    hend;
  int*    vstart;
  int*    vend;

  const auto&         top_cell = db.cells.back();
  std::vector<h_edge> hes1;
  std::vector<h_edge> hes2;
  std::vector<v_edge> ves1;
  std::vector<v_edge> ves2;
  for (int i = 0; i < top_cell.cell_refs.size(); ++i) {
    const auto& cell_ref = top_cell.cell_refs.at(i);
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (!the_cell.is_touching(layer1) and !the_cell.is_touching(layer2)) {
      continue;
    }
    if (the_cell.is_touching(layer1)) {
      hes1.insert(hes1.end(), top_cell.cell_refs[i].h_edges1.begin(),
                  top_cell.cell_refs[i].h_edges1.end());
      ves1.insert(ves1.end(), top_cell.cell_refs[i].v_edges1.begin(),
                  top_cell.cell_refs[i].v_edges1.end());
    }
    if (the_cell.is_touching(layer2)) {
      hes2.insert(hes2.end(), top_cell.cell_refs[i].h_edges2.begin(),
                  top_cell.cell_refs[i].h_edges2.end());
      ves2.insert(ves2.end(), top_cell.cell_refs[i].v_edges2.begin(),
                  top_cell.cell_refs[i].v_edges2.end());
    }
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "before: " << cudaGetErrorString(err) << std::endl;
  }
  cudaMalloc((void**)&dhes1, sizeof(h_edge) * hes1.size());
  cudaMalloc((void**)&dves1, sizeof(v_edge) * ves1.size());
  cudaMalloc((void**)&dhes2, sizeof(h_edge) * hes2.size());
  cudaMalloc((void**)&dves2, sizeof(v_edge) * ves2.size());
  cudaMalloc((void**)&hstart, sizeof(int) * hes2.size());
  cudaMalloc((void**)&hend, sizeof(int) * hes2.size());
  cudaMalloc((void**)&vstart, sizeof(int) * ves2.size());
  cudaMalloc((void**)&vend, sizeof(int) * ves2.size());
  cudaMalloc((void**)&dres, sizeof(check_result) * 100000);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "init:" << cudaGetErrorString(err) << std::endl;
    return;
  }
  cudaMemcpyAsync(dhes1, hes1.data(), sizeof(h_edge) * hes1.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(dves1, ves1.data(), sizeof(v_edge) * ves1.size(),
                  cudaMemcpyHostToDevice, stream2);
  cudaMemcpyAsync(dhes2, hes2.data(), sizeof(h_edge) * hes2.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(dves2, ves2.data(), sizeof(v_edge) * ves2.size(),
                  cudaMemcpyHostToDevice, stream2);

  cudaDeviceSynchronize();

  thrust::async::sort(
      thrust::device, dhes1, dhes1 + hes1.size(),
      [] __device__(h_edge & h1, h_edge & h2) { return h1.y < h2.y; });
  thrust::async::sort(
      thrust::device, dves1, dves1 + ves1.size(),
      [] __device__(v_edge & v1, v_edge & v2) { return v1.x < v2.x; });
  thrust::async::sort(
      thrust::device, dhes2, dhes2 + hes2.size(),
      [] __device__(h_edge & h1, h_edge & h2) { return h1.y < h2.y; });
  thrust::async::sort(
      thrust::device, dves2, dves2 + ves2.size(),
      [] __device__(v_edge & v1, v_edge & v2) { return v1.x < v2.x; });

  cudaDeviceSynchronize();
  int bs = 256;
  hori_search<<<(hes2.size() + bs - 1) / bs, bs, 0, stream1>>>(
      dhes1, dhes2, hes1.size(), hes2.size(), threshold, hstart, hend);
  vert_search<<<(ves2.size() + bs - 1) / bs, bs, 0, stream2>>>(
      dves1, dves2, ves1.size(), ves2.size(), threshold, vstart, hend);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "search: " << cudaGetErrorString(err) << std::endl;
  }
  hori_check<<<(hes2.size() + bs - 1) / bs, bs, 0, stream1>>>(
      hstart, hend, dhes1, dhes2, hes2.size(), threshold, dres);
  vert_check<<<(ves2.size() + bs - 1) / bs, bs, 0, stream2>>>(
      vstart, vend, dves1, dves2, ves2.size(), threshold, dres);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "final: " << cudaGetErrorString(err) << std::endl;
  }
  cudaFree(dhes1);
  cudaFree(dhes2);
  cudaFree(dves1);
  cudaFree(dves2);
  cudaFree(hstart);
  cudaFree(hend);
  cudaFree(vstart);
  cudaFree(vend);
  cudaFree(dres);
  t.pause();
}
}  // namespace odrc