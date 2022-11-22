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

__global__ void hori_search(h_edge* hedges,
                            int     size,
                            int     threshold,
                            int*    estarts) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  int l   = 0;
  int r   = tid - 1;
  int y   = hedges[tid].y;
  int mid = tid;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_y = hedges[mid].y;
    if (y - mid_y >= threshold) {
      l = mid + 1;
    } else {
      r = mid - 1;
    }
  }
  estarts[tid] = tid - mid;
}

__global__ void vert_search(v_edge* vedges,
                            int     size,
                            int     threshold,
                            int*    estarts) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  int l   = 0;
  int r   = tid - 1;
  int x   = vedges[tid].x;
  int res = l;
  int mid = tid;
  while (l <= r) {
    mid       = (l + r) / 2;
    int mid_x = vedges[mid].x;
    if (x - mid_x >= threshold) {
      l = mid + 1;
    } else {
      r = mid - 1;
    }
  }
  estarts[tid] = tid - mid;
}

__global__ void hori_check(int*          estarts,
                           h_edge*       hedges,
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
  h_edge& e    = hedges[tid];
  int     e11x = e.x1;
  int     e11y = e.y;
  int     e12x = e.x2;
  int     e12y = e.y;
  for (int i = tid - estarts[tid]; i < tid; ++i) {
    h_edge& ee           = hedges[i];
    int     e21x         = ee.x1;
    int     e21y         = ee.y;
    int     e22x         = ee.x2;
    int     e22y         = ee.y;
    bool    is_violation = false;
    // if (e11y - e21y >= threshold)
    //   break;

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
      if (offset < tid * 2 + 2 and offset < 2000000) {
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
                           v_edge*       vedges,
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
  auto& e      = vedges[tid];
  int   e11x   = e.x;
  int   e11y   = e.y1;
  int   e12x   = e.x;
  int   e12y   = e.y2;
  for (int i = tid - estarts[tid]; i < tid; ++i) {
    v_edge& ee           = vedges[i];
    int     e21x         = ee.x;
    int     e21y         = ee.y1;
    int     e22x         = ee.x;
    int     e22y         = ee.y2;
    bool    is_violation = false;

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
      if (offset < tid * 2 + 2 and offset < 2000000) {
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

void space_check_dac23(const odrc::core::database& db,
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
  h_edge* dhes;
  v_edge* dves;
  int*    hstart;
  int*    vstart;
  cudaMallocAsync((void**)&dres, sizeof(check_result) * 10000000, stream1);

  const auto&         top_cell = db.cells.back();
  std::vector<h_edge> hes;
  std::vector<v_edge> ves;
  for (int i = 0; i < top_cell.cell_refs.size(); ++i) {
    const auto& cell_ref = top_cell.cell_refs.at(i);
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (!the_cell.is_touching(layer1) and !the_cell.is_touching(layer2)) {
      continue;
    }
    hes.insert(hes.end(), top_cell.cell_refs[i].h_edges1.begin(),
               top_cell.cell_refs[i].h_edges1.end());
    ves.insert(ves.end(), top_cell.cell_refs[i].v_edges1.begin(),
               top_cell.cell_refs[i].v_edges1.end());
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
  }
  cudaMalloc((void**)&dhes, sizeof(h_edge) * hes.size());
  cudaMalloc((void**)&dves, sizeof(v_edge) * ves.size());
  cudaMalloc((void**)&hstart, sizeof(int) * hes.size());
  cudaMalloc((void**)&vstart, sizeof(int) * ves.size());
  cudaMemcpyAsync(dhes, hes.data(), sizeof(h_edge) * hes.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(dves, ves.data(), sizeof(v_edge) * ves.size(),
                  cudaMemcpyHostToDevice, stream2);

  thrust::async::sort(
      thrust::device, dhes, dhes + hes.size(),
      [] __device__(h_edge & h1, h_edge & h2) { return h1.y < h2.y; });
  thrust::async::sort(
      thrust::device, dves, dves + ves.size(),
      [] __device__(v_edge & v1, v_edge & v2) { return v1.x < v2.x; });

  int bs = 256;
  hori_search<<<(hes.size() + bs - 1) / bs, bs, 0, stream1>>>(
      dhes, hes.size(), threshold, hstart);
  vert_search<<<(ves.size() + bs - 1) / bs, bs, 0, stream2>>>(
      dves, ves.size(), threshold, vstart);
  hori_check<<<(hes.size() + bs - 1) / bs, bs, 0, stream1>>>(
      hstart, dhes, hes.size(), threshold, dres);
  vert_check<<<(ves.size() + bs - 1) / bs, bs, 0, stream2>>>(
      vstart, dves, hes.size(), threshold, dres);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
  }
  t.pause();
}
}  // namespace odrc