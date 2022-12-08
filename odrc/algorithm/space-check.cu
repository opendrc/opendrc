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

namespace odrc {

using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;
using odrc::core::h_edge;
using odrc::core::v_edge;

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
__global__ void init(h_edge* array, int* keys, int* indices, int size) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  for (int i = tid; i < size; i += blockDim.x) {
    keys[i]    = array[tid].y;
    indices[i] = tid;
  }
}

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
  int res = l;
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
  int offset = tid * 10;
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
    if (is_violation and false) {
      if (offset < tid * 10 + 10) {
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
  int   offset = tid * 10;
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
      if (offset < tid * 10 + 10) {
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

__global__ void hori_check_bak(int*          estarts,
                               h_edge*       hedges,
                               int           size,
                               int           threshold,
                               check_result* results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  __shared__ int vio_offset;
  if (tid == 0) {
    vio_offset = blockIdx.x * 1000;
  }
  __syncthreads();
  // printf("hello");
  int l = 0;
  int r = size - 1;
  int mid;
  while (l <= r) {
    mid     = (l + r) / 2;
    int sum = estarts[mid];
    if (sum < tid) {
      l = mid + 1;
    } else {
      r = mid - 1;
    }
  }
  if (estarts[mid] < tid)
    ++mid;
  int     d    = estarts[mid] - tid + 1;
  h_edge& e    = hedges[mid];
  int     e11x = e.x1;
  int     e11y = e.y;
  int     e12x = e.x2;
  int     e12y = e.y;
  for (int i = tid; i > 0; --i) {
    h_edge& ee           = hedges[mid - d];
    int     e21x         = ee.x1;
    int     e21y         = ee.y;
    int     e22x         = ee.x2;
    int     e22y         = ee.y;
    bool    is_violation = false;
    if (e11y - e21y >= threshold)
      break;

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
      int offset = atomicAdd(&vio_offset, 1);
      if (offset < (blockIdx.x + 1) * 1000 and offset < 90000) {
        check_result& res = results[offset];
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
  }
}

void space_check_pal(const odrc::core::database& db,
                     int                         layer1,
                     int                         layer2,
                     int                         threshold,
                     std::vector<check_result>&  vios) {
  const auto&             cell_refs = db.cells.back().cell_refs;
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
    y.insert(cr.mbr1[2]);
    y.insert(cr.mbr1[3]);
    lrs.emplace_back(cr.mbr1[2]);
    lrs.emplace_back(cr.mbr1[3]);
    hidx.emplace_back(hes.size());
    vidx.emplace_back(ves.size());
    hes.insert(hes.end(), cr.h_edges.begin(), cr.h_edges.end());
    ves.insert(ves.end(), cr.v_edges.begin(), cr.v_edges.end());
    mbrs.emplace_back(cr.mbr1[0]);
    mbrs.emplace_back(cr.mbr1[1]);
    mbrs.emplace_back(cr.mbr1[2]);
    mbrs.emplace_back(cr.mbr1[3]);
  }
  hidx.emplace_back(hes.size());
  vidx.emplace_back(ves.size());
  cudaError_t  err;
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  err = cudaGetLastError();
  // std::cout << cudaGetErrorString(err) << std::endl;

  // std::cout << "loop through: " << t.get_elapsed() << std::endl;
  // std::cout << "# cells: " << cells.size() << std::endl;
  // std::cout << "hidx.size(): " << hidx.size() << std::endl;

  h_edge*       h_edges;
  v_edge*       v_edges;
  int*          h_idx;
  int*          v_idx;
  int*          cells_d;
  int*          mbrs_d;
  check_result* results;

  // cudaMallocAsync((void**)&h_edges, sizeof(h_edge) * hes.size(),
  // stream1); cudaMallocAsync((void**)&v_edges, sizeof(v_edge) *
  // ves.size(), stream1); cudaMemcpyAsync(h_edges, hes.data(),
  // sizeof(h_edge) * hes.size(),
  //                 cudaMemcpyHostToDevice, stream1);
  // cudaMemcpyAsync(v_edges, ves.data(), sizeof(v_edge) * ves.size(),
  //                 cudaMemcpyHostToDevice, stream1);
  // cudaMallocAsync((void**)&h_idx, sizeof(int) * hidx.size(), stream1);
  // cudaMallocAsync((void**)&v_idx, sizeof(int) * vidx.size(), stream1);
  // cudaMemcpyAsync(h_idx, hidx.data(), sizeof(int) * hidx.size(),
  //                 cudaMemcpyHostToDevice, stream1);
  // cudaMemcpyAsync(v_idx, vidx.data(), sizeof(int) * vidx.size(),
  //                 cudaMemcpyHostToDevice, stream1);
  // cudaMallocAsync((void**)&mbrs_d, sizeof(int) * mbrs.size(), stream1);
  // cudaMemcpyAsync(mbrs_d, mbrs.data(), sizeof(int) * mbrs.size(),
  //                 cudaMemcpyHostToDevice, stream1);
  cudaMallocAsync((void**)&cells_d, sizeof(int) * cells.size(), stream1);
  cudaMemcpyAsync(cells_d, cells.data(), sizeof(int) * cells.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMallocAsync((void**)&results, sizeof(check_result) * 200000, stream1);
  h_edge* rhes_d;
  v_edge* rves_d;
  cudaMallocAsync((void**)&rhes_d, sizeof(h_edge) * 5000 * 100, stream1);
  cudaMallocAsync((void**)&rves_d, sizeof(v_edge) * 5000 * 100, stream1);
  std::vector<int> yv(y.begin(), y.end());
  std::sort(yv.begin(), yv.end());
  std::vector<int> y_comp(yv.back() + 5);
  // std::cout << "UNIQUE Y: " << yv.size() << std::endl;
  // std::cout << yv.back() << std::endl;
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
  // std::cout << "# rows: " << rows.size() << std::endl;
  for (int i = 0; i < csize; ++i) {
    rows[ufv[lrs[i * 2]]].emplace_back(i);
  }
  // t.pause();
  // std::cout << "UF and data transfer: " << t.get_elapsed() << std::endl;
  // std::vector<event> events;
  // std::vector<int>   eidx;
  // events.reserve(cells.size() * 2);
  // eidx.reserve(rows.size() + 1);
  // eidx.emplace_back(0);
  // event* events_d = nullptr;
  // int*   eidx_d   = nullptr;
  // cudaMallocAsync((void**)&events_d, cells.size() * sizeof(event) * 2,
  // stream1); cudaMallocAsync((void**)&eidx_d, (rows.size() + 1) *
  // sizeof(int), stream1); thrust::device_vector<int>    keysv(5000 * 100);
  // thrust::device_vector<int>    indicesv(5000 * 100);
  // thrust::device_vector<h_edge> rhesv(5000 * 100);
  int bs = 128;
  for (int i = 0; i < rows.size(); ++i) {
    int bs          = 128;
    int rsize       = rows[i].size();
    int total_check = rsize * (rsize + 1) / 2;

    std::vector<h_edge> rhes;
    std::vector<v_edge> rves;

    // int start_offset = events.size();
    int sync_start = 0;
    int sync_end   = 2048;
    for (int j = 0; j < rows[i].size(); ++j) {
      int cid = cells.at(rows[i][j]);
      rhes.insert(rhes.end(), cell_refs.at(cid).h_edges.begin(),
                  cell_refs.at(cid).h_edges.end());
      rves.insert(rves.end(), cell_refs.at(cid).v_edges.begin(),
                  cell_refs.at(cid).v_edges.end());
      // if (rhes.size() >= sync_end) {
      //   if (rhes.size() >= 5000 * 100) {
      //     assert(false);
      //   }
      //   cudaMemcpyAsync(rhes_d + sync_start, &rhes[sync_start],
      //                   sizeof(h_edge) * (rhes.size() - sync_start),
      //                   cudaMemcpyHostToDevice, stream1);
      //   cudaMemcpyAsync(rves_d + sync_start, &rves[sync_start],
      //                   sizeof(v_edge) * (rves.size() - sync_start),
      //                   cudaMemcpyHostToDevice, stream1);
      //   sync_start = sync_end;
      //   sync_end += 1024;
      // }
    }
    int* estarts  = nullptr;
    int* estarts2 = nullptr;
    if (rhes.size() < 1000) {
      std::sort(rhes.begin(), rhes.end(),
                [] __device__(const h_edge& h1, const h_edge& h2) {
                  return h1.y < h2.y;
                });
      std::sort(rves.begin(), rves.end(),
                [] __device__(const v_edge& v1, const v_edge& v2) {
                  return v1.x < v2.x;
                });
      /*
      cudaMemcpyAsync(rhes_d, rhes.data(), sizeof(h_edge) * rhes.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaMemcpyAsync(rves_d, rves.data(), sizeof(v_edge) * rves.size(),
                      cudaMemcpyHostToDevice, stream2);
                      */
      for (int j = 0; j < rves.size(); ++j) {
        auto& e    = rves[j];
        int   e11x = e.x;
        int   e11y = e.y1;
        int   e12x = e.x;
        int   e12y = e.y2;
        for (int k = j - 1; k >= 0; --k) {
          if (rves[j].x - rves[k].x >= threshold)
            break;
          v_edge& ee           = rves[k];
          int     e21x         = ee.x;
          int     e21y         = ee.y1;
          int     e22x         = ee.x;
          int     e22y         = ee.y2;
          bool    is_violation = false;
          if (e11y - e21y >= threshold)
            break;

          if (e11x < e21x) {
            // e11 e22
            // e12 e21
            bool is_outside_to_outside = e11y > e12y and e21y < e22y;
            bool is_too_close          = e21x - e11x < threshold;
            bool is_projection_overlap = e11y < e21y and e22y < e12y;
            is_violation = is_outside_to_outside and is_too_close and
                           is_projection_overlap;

          } else {
            // e21 e12
            // e22 e11
            bool is_outside_to_outside = e21y > e22y and e11y < e21y;
            bool is_too_close          = e11x - e21x < threshold;
            bool is_projection_overlap = e21y < e11y and e12y < e22y;
            is_violation = is_outside_to_outside and is_too_close and
                           is_projection_overlap;
          }
          if (is_violation) {
            vios.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x, e21y,
                                           e22x, e22y, true});
          }
        }
      }

      for (int j = 0; j < rhes.size(); ++j) {
        auto& e    = rhes[j];
        int   e11x = e.x1;
        int   e11y = e.y;
        int   e12x = e.x2;
        int   e12y = e.y;
        for (int k = j - 1; k >= 0; --k) {
          if (rhes[j].y - rhes[k].y >= threshold)
            break;
          h_edge& ee           = rhes[k];
          int     e21x         = ee.x1;
          int     e21y         = ee.y;
          int     e22x         = ee.x2;
          int     e22y         = ee.y;
          bool    is_violation = false;
          if (e11y - e21y >= threshold)
            break;

          if (e11y < e22y) {
            // e22 e21
            // e11 e12
            bool is_outside_to_outside = e11x < e12x and e21x > e22x;
            bool is_too_close          = e21y - e11y < threshold;
            bool is_projection_overlap = e21x < e11x and e12x < e22x;
            is_violation = is_outside_to_outside and is_too_close and
                           is_projection_overlap;
          } else {
            // e12 e11
            // e21 e22
            bool is_outside_to_outside = e21x < e22x and e11x > e12x;
            bool is_too_close          = e11y - e21y < threshold;
            bool is_projection_overlap = e11x < e21x and e22x < e12x;
            is_violation = is_outside_to_outside and is_too_close and
                           is_projection_overlap;
          }
          if (is_violation) {
            vios.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x, e21y,
                                           e22x, e22y, true});
          }
        }
      }

      continue;
    } else {
      cudaMallocAsync((void**)&estarts, sizeof(int) * rhes.size(), stream1);
      cudaMallocAsync((void**)&estarts2, sizeof(int) * rhes.size(), stream2);
      cudaMemcpyAsync(rhes_d, rhes.data(), sizeof(h_edge) * rhes.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaMemcpyAsync(rves_d, rves.data(), sizeof(v_edge) * rves.size(),
                      cudaMemcpyHostToDevice, stream2);
      cudaStreamSynchronize(stream1);
      thrust::async::sort(thrust::device, rhes_d, rhes_d + rhes.size(),
                          [] __device__(const h_edge& h1, const h_edge& h2) {
                            return h1.y < h2.y;
                          });
      cudaStreamSynchronize(stream2);
      thrust::async::sort(thrust::device, rves_d, rves_d + rves.size(),
                          [] __device__(const v_edge& v1, const v_edge& v2) {
                            return v1.x < v2.x;
                          });
    }
    cudaDeviceSynchronize();

    // h2.start();
    // cudaMemcpyAsync(rhes_d, rhes.data(), sizeof(h_edge) * rhes.size(),
    //                 cudaMemcpyHostToDevice, stream1);
    // cudaMemcpyAsync(rves_d, rves.data(), sizeof(v_edge) * rves.size(),
    //                 cudaMemcpyHostToDevice, stream1);
    // cudaStreamSynchronize(stream1);
    // cudaMemcpy(rves_d, rves.data(), sizeof(v_edge) * rves.size(),
    //            cudaMemcpyHostToDevice);
    // std::cout << "sort size: " << rhes.size() << std::endl;
    // h4.start();
    // h3.start();
    // h3.pause();
    // thrust::async::sort(thrust::device, thrust::device, rves_d,
    //                     rves_d + rves.size(),
    //                     [] __device__(const v_edge& v1, const v_edge& v2)
    //                     {
    //                       return v1.x < v2.x;
    //                     });
    // auto nosync1 = thrust::cuda::par_nosync.on(stream1);
    // auto nosync2 = thrust::cuda::par_nosync.on(stream2);
    // thrust::sort(nosync1, rhes_d, rhes_d + rhes.size(),
    //              [] __device__(const h_edge& h1, const h_edge& h2) {
    //                return h1.y < h2.y;
    //              });
    // h3.pause();

    // h4.pause();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << cudaGetErrorString(err) << std::endl;
    }
    // h2.pause();
    // int end_offset = events.size();
    // eidx.emplace_back(end_offset);
    // cudaStreamSynchronize(stream1);
    // cudaMemcpyAsync(events_d + start_offset, &events[start_offset],
    //                 sizeof(event) * (end_offset - start_offset),
    //                 cudaMemcpyHostToDevice, stream1);
    // event* array = thrust::raw_pointer_cast(events_d.data());
    // int    N     = events.size();
    // auto   serr  = cudaGetLastError();
    // int     N   = rhes.size();
    // h_edge* tmp = thrust::raw_pointer_cast(rhesv.data());
    // if (rhes.size() > 8000) {
    //   h3.start();
    //   h_edge* array   = rhes_d;
    //   int*    keys    = thrust::raw_pointer_cast(keysv.data());
    //   int*    indices = thrust::raw_pointer_cast(indicesv.data());
    //   bs              = 1024;
    //   init<<<1, bs, 0, stream1>>>(array, keys, indices, N);
    //   cudaStreamSynchronize(stream1);
    //   err = cudaGetLastError();
    //   if (err != cudaSuccess) {
    //     std::cout << "init failed: " << cudaGetErrorString(err) <<
    //     std::endl;
    //   }
    //   thrust::sort_by_key(thrust::device, keys, keys + N, indices);
    //   thrust::copy_n(thrust::device,
    //                  thrust::make_permutation_iterator(array, indices),
    //                  N, tmp);
    //   h3.pause();
    // }
    // if (true) {
    //   if (rhes.size() > 8000) {
    //     h4.start();
    //   }
    //   if (rhes.size() > 8000) {
    //     h4.pause();
    //   }
    // }
    // cudaStreamSynchronize(stream1);
    // h3.start();
    // thrust::async::sort(thrust::device, rhes_d, rhes_d + rhes.size(),
    //                     [] __device__(const h_edge& h1, const h_edge& h2)
    //                     {
    //                       return h1.y < h2.y;
    //                     });
    // thrust::async::sort(thrust::device, rves_d, rves_d + rves.size(),
    //                     [] __device__(const v_edge& v1, const v_edge& v2)
    //                     {
    //                       return v1.x < v2.x;
    //                     });

    // std::sort(events.begin(), events.end(), [](const auto& e1, const
    // auto& e2) {
    //   return e1.x == e2.x ? e1.id > e2.id : e1.x < e2.x;
    // });
    // h3.pause();
    // h2.pause();
    bs = 128;
    hori_search<<<(rhes.size() + bs - 1) / bs, bs, 0, stream1>>>(
        rhes_d, rhes.size(), threshold, estarts);
    vert_search<<<(rves.size() + bs - 1) / bs, bs, 0, stream2>>>(
        rves_d, rves.size(), threshold, estarts2);
    // std::vector<int> es(rhes.size());
    // cudaMemcpy(es.data(), estarts, sizeof(int)*rhes.size(),
    // cudaMemcpyDeviceToHost);
    // int ess = 0; int mes = 0; for(int a: es) {
    //   mes = std::max(mes, a);
    //   ess += a;
    // }
    // std::cout<< "max: " << mes << ", avg: " << ess/double(es.size()) <<
    // std::endl;
    // thrust::inclusive_scan(thrust::device, estarts, estarts +
    // rhes.size(),
    //                        estarts);
    // int total_checks;
    // cudaMemcpy(&total_checks, estarts + rhes.size() - 1, sizeof(int),
    //            cudaMemcpyDeviceToHost);
    bs  = 128;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << cudaGetErrorString(err) << std::endl;
    }
    cudaStreamSynchronize(stream1);
    hori_check<<<(rhes.size() + bs - 1) / bs, bs, 0, stream1>>>(
        estarts, rhes_d, rhes.size(), threshold, results);
    cudaStreamSynchronize(stream2);
    vert_check<<<(rves.size() + bs - 1) / bs, bs, 0, stream2>>>(
        estarts2, rves_d, rhes.size(), threshold, results);

    // hori_check<<<1, 1, 0, stream1>>>(
    //     estarts, thrust::raw_pointer_cast(rhes_d.data()), rhes.size(),
    //     threshold, results);
  }
  check_result rlts[200000];
  cudaMemcpyAsync(rlts, &results, sizeof(check_result) * 200000,
                  cudaMemcpyDeviceToHost);
  for (int i = 0; i < 200000; i++) {
    if (rlts[i].is_violation) {
      vios.emplace_back(rlts[i]);
    }
  }
  return;
}
}  // namespace odrc