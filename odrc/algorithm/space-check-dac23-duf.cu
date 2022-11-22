#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <thrust/device_vector.h>
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

__global__ void run_check(h_edge*       h_edges,
                          v_edge*       v_edges,
                          int*          h_idx,
                          int*          v_idx,
                          int*          mbrs_d,
                          check_result* results,
                          int*          cells,
                          int           csize,
                          int           threshold,
                          int           total) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= csize) {
    return;
  }
  int sum = 0;
  int i   = cells[tid * 2];      // cell1
  int j   = cells[tid * 2 + 1];  // cell2

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

  // horizontal checks
  int his = h_idx[i];
  int hie = h_idx[i + 1];
  int hjs = h_idx[j];
  int hje = h_idx[j + 1];
  for (int i = his; i < hie; ++i) {
    int iy = h_edges[i].y;
    if (iy - threshold >= jymax)
      break;
    else if (iy + threshold <= jymin)
      continue;
    for (int j = hjs; j < hje; ++j) {
      int jy = h_edges[j].y;
      if (jy - threshold >= iy)
        break;
      else if (jy + threshold <= iy)
        continue;
      check_result& res = results[tid];

      int  e11x         = h_edges[i].x1;
      int  e11y         = iy;
      int  e12x         = h_edges[i].x2;
      int  e12y         = iy;
      int  e21x         = h_edges[j].x1;
      int  e21y         = jy;
      int  e22x         = h_edges[j].x2;
      int  e22y         = jy;
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
        check_result& res = results[tid];
        res.e11x          = h_edges[i].x1;
        res.e11y          = iy;
        res.e12x          = h_edges[i].x2;
        res.e12y          = iy;
        res.e21x          = h_edges[j].x1;
        res.e21y          = jy;
        res.e22x          = h_edges[j].x2;
        res.e22y          = jy;
        res.is_violation  = false;
      }
    }
  }

  // vertical check
  int vis = v_idx[i];
  int vie = v_idx[i + 1];
  int vjs = v_idx[j];
  int vje = v_idx[j + 1];
  for (int i = vis; i < vie; ++i) {
    int ix = v_edges[i].x;
    if (ix - threshold >= jxmax)
      break;
    else if (ix + threshold <= jxmin)
      continue;
    for (int j = vjs; j < vje; ++j) {
      int jx = v_edges[j].x;
      if (jx - threshold >= ix)
        break;
      else if (jx + threshold <= ix)
        continue;
      check_result& res          = results[tid];
      int           e11x         = ix;
      int           e11y         = v_edges[i].y1;
      int           e12x         = ix;
      int           e12y         = v_edges[i].y2;
      int           e21x         = jx;
      int           e21y         = v_edges[j].y1;
      int           e22x         = jx;
      int           e22y         = v_edges[j].y2;
      bool          is_violation = false;

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
        check_result& res = results[tid];
        res.e11x          = ix;
        res.e11y          = v_edges[i].y1;
        res.e12x          = ix;
        res.e12y          = v_edges[i].y2;
        res.e21x          = jx;
        res.e21y          = v_edges[j].y1;
        res.e22x          = jx;
        res.e22y          = v_edges[j].y2;
        res.is_violation  = true;
      }
    }
  }
}

__global__ void comp(int* mbrs, int* xs, int size, int* xcomp) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  xcomp[mbrs[xs[tid] * 4]]     = 1;
  xcomp[mbrs[xs[tid] * 4 + 1]] = 1;
}

__global__ void init(int* uf, int size) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  uf[tid] = tid;
}

__global__ void fill(int* uf, int* mbrs, int* xs, int size, int* xcomp) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  int l = xcomp[mbrs[xs[tid] * 4]];
  int r = xcomp[mbrs[xs[tid] * 4 + 1]];
  if (r - l < 20) {
    atomicMax(uf + l, r);
  }
}

__global__ void unionuf(int* uf, int size) {
  int end  = -1;
  int lidx = -1;
  for (int i = 0; i < size; ++i) {
    if (i > end) {
      end = uf[i];
      ++lidx;
    }
    end   = end > uf[i] ? end : uf[i];
    uf[i] = lidx;
  }
}

__global__ void assign(int* uf,
                       int* mbrs,
                       int* xs,
                       int  size,
                       int* xcomp,
                       int* group) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size) {
    return;
  }
  int l = xcomp[mbrs[xs[tid] * 4]];
  int r = xcomp[mbrs[xs[tid] * 4 + 1]];
  if (r - l < 20) {
    group[tid] = uf[l];
  } else {
    group[tid] = -1;  // long object
  }
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
  int lidx  = -1;
  int label = 0;

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
  t.start();
  std::vector<int> x_comp(1000000);
  for (int i = 0; i < rows.size(); ++i) {
    loop.start();
    int bs          = 128;
    int rsize       = rows[i].size();
    int total_check = rsize * (rsize + 1) / 2;
    // std::vector<int> checks;
    // std::cout << "row " << i << ", rsize=" << rsize
    //           << ", total check = " << checks.size() / 2 << "/" <<
    //           total_check
    //           << std::endl;
    // std::unordered_set<int> x;
    // x.reserve(rows[i].size() * 2);
    std::vector<int> rlrs;
    rlrs.reserve(rows[i].size() * 2);
    loop.start();
    h1.start();
    int max_x = 0;
    // /*
    for (int j = 0; j < rows[i].size(); ++j) {
      int cid = rows[i][j];
      // x.emplace(mbrs[cid * 4]);
      // x.emplace(mbrs[cid * 4 + 1]);
      // x_comp[mbrs[cid * 4]]     = 1;
      // x_comp[mbrs[cid * 4 + 1]] = 1;
      max_x = std::max(max_x, mbrs[cid * 4 + 1]);
      rlrs.emplace_back(mbrs[cid * 4]);
      rlrs.emplace_back(mbrs[cid * 4 + 1]);
    }
    // */
    thrust::device_vector<int> dv(max_x + 5, 0);
    thrust::device_vector<int> dri(rows[i]);
    // thrust::device_vector<int> dsx(rlrs);
    int nbs = 128;
    comp<<<(rows[i].size() + nbs - 1) / nbs, nbs>>>(
        mbrs_d, thrust::raw_pointer_cast(dri.data()), rows[i].size(),
        thrust::raw_pointer_cast(dv.data()));
    cudaDeviceSynchronize();
    // std::cout << rlrs.size() << " " << rows[i].size() << " " << dv.size() <<
    // std::endl;
    // assert(rlrs.size() == rows[i].size() * 2);
    thrust::inclusive_scan(dv.begin(), dv.end(), dv.begin());
    // thrust::copy(dv.begin(), dv.end(), x_comp.begin());
    int uf_size = dv.back() + 5;
    // std::cout << "uf_size: " << uf_size << std::endl;
    thrust::device_vector<int> duf(uf_size, 0);
    // std::cout << "duf: " << duf.size() << std::endl;
    int* dufp = thrust::raw_pointer_cast(duf.data());
    init<<<(uf_size + nbs - 1) / nbs, nbs>>>(dufp, uf_size);
    auto init_err = cudaDeviceSynchronize();
    // std::cout << "init res: " << cudaGetErrorString(init_err) << std::endl;
    assert(init_err == 0);
    fill<<<(rows[i].size() + nbs - 1) / nbs, nbs>>>(
        dufp, mbrs_d, thrust::raw_pointer_cast(dri.data()), rows[i].size(),
        thrust::raw_pointer_cast(dv.data()));
    auto fill_err = cudaDeviceSynchronize();
    unionuf<<<1, 1>>>(dufp, uf_size);
    auto unionuf_err = cudaDeviceSynchronize();
#if false
    std::vector<int> cduf(uf_size);
    thrust::copy(duf.begin(), duf.end(), cduf.begin());
    for(int i = 0; i < 100; ++i) {
      std::cout << cduf[i] << " ";
    }
    std::cout << std::endl;
    return;
#endif
    // std::cout << "unionuf res: " << cudaGetErrorString(unionuf_err)
    //           << std::endl;
    assert(fill_err == 0);
    loop.pause();
    thrust::device_vector<int> groups(rows[i].size(), 0);
    assign<<<(rows[i].size() + nbs - 1) / nbs, nbs>>>(
        dufp, mbrs_d, thrust::raw_pointer_cast(dri.data()), rows[i].size(),
        thrust::raw_pointer_cast(dv.data()),
        thrust::raw_pointer_cast(groups.data()));
    auto assign_err = cudaDeviceSynchronize();
    // std::cout << "assign res: " << cudaGetErrorString(assign_err) <<
    // std::endl;
    std::vector<int> groups_host(rows[i].size(), 0);
    // assert(fill_err == 0);
    thrust::copy(groups.begin(), groups.end(), groups_host.begin());

    std::vector<std::vector<int>> columns(uf_size + 1);
    std::vector<int>              long_objects;
    for (int ii = 0; ii < rows[i].size(); ++ii) {
      if (groups[ii] == -1) {
        long_objects.emplace_back(rows[i][ii]);
      } else {
        columns[groups[ii]].emplace_back(rows[i][ii]);
      }
    }
    std::vector<int> checks;
    for (int ii = 0; ii < long_objects.size(); ++ii) {
      for (int j = 0; j < rows[i].size(); ++j) {
        checks.emplace_back(long_objects[ii]);
        checks.emplace_back(rows[i][j]);
      }
    }
    std::cout << "#columns: " << columns.size() << std::endl;
    std::cout << "#long objects: " << long_objects.size() << std::endl;
    for (int ii = 0; ii < columns.size(); ++ii) {
      for (int j = 0; j < columns[ii].size(); ++j) {
        for (int k = j; k < columns[ii].size(); ++k) {
          checks.emplace_back(columns[ii][j]);
          checks.emplace_back(columns[ii][k]);
        }
      }
    }
    std::cout << "# checks:" << checks.size() / 2 << std::endl;

    // cudaMemcpy(cells_d, checks.data(), sizeof(int) * checks.size(),
    //            cudaMemcpyHostToDevice);
    thrust::device_vector<int> checks_d(checks);
    run_check<<<(checks.size() / 2 + bs - 1) / bs, bs>>>(
        h_edges, v_edges, h_idx, v_idx, mbrs_d, results,
        thrust::raw_pointer_cast(checks_d.data()), checks.size() / 2, threshold,
        cells.size());
    auto check_err = cudaDeviceSynchronize();
    assert(check_err == 0);
    continue;

#if false
    // std::vector<int> xv(x.begin(), x.end());
    // std::sort(xv.begin(), xv.end());
    // if (xv.back() + 5 >= x_comp.size()) {
    // x_comp.resize(xv.back() + 5);
    // }
    // std::cout << "max x:" << xv.back() << std::endl;
    int discrete = 0;
    // for (int ii = 0; ii < xv.size(); ++ii) {
    //   x_comp[xv[ii]] = ii;
    // }
    // h2.start();
    // for (int ii = 0; ii < max_x + 3; ++ii) {
    //   x_comp[ii] = x_comp[ii] == 0 ? 0 : discrete++;
    // }
    // h2.pause();
    // std::cout << max_x << " " << discrete << std::endl;
    // std::cout << "distinct x: " << xv.size() << std::endl;
    for (int ii = 0; ii < rlrs.size(); ++ii) {
      rlrs[ii] = x_comp[rlrs[ii]];
    }
    loop.pause();
    // continue;
    // std::vector<int> rufv(xv.size(), 0);
    std::vector<int> rufv(x_comp[max_x], 0);
    std::iota(rufv.begin(), rufv.end(), 0);

    int              rlrs_size = rlrs.size();
    std::vector<int> long_objs;
    for (int ii = 0; ii < rlrs_size; ii += 2) {
      int ufb = rlrs[ii];
      int ufu = rlrs[ii + 1];
      if (ufu - ufb > 3) {
        long_objs.emplace_back(ii);
      } else {
        rufv[ufb] = rufv[ufb] > ufu ? rufv[ufb] : ufu;
      }
    }
    // std::cout << "#long objects: " << long_objs.size() << std::endl;
    int rlidx = -1;

    int rend = -1;
    for (int ii = 0; ii < rufv.size(); ++ii) {
      if (ii > rend) {
        rend = rufv[ii];
        ++rlidx;
      }
      rend     = std::max(rend, rufv[ii]);
      rufv[ii] = rlidx;
    }
    std::vector<std::vector<int>> columns(rlidx + 1);
    std::cout << "# columns: " << columns.size() << std::endl;
    // loop.pause();
    continue;
    for (int ii = 0; ii < csize; ++ii) {
      columns[rufv[rlrs[ii * 2]]].emplace_back(ii);
    }

    cudaMemcpy(cells_d, checks.data(), sizeof(int) * checks.size(),
               cudaMemcpyHostToDevice);
    run_check<<<(checks.size() / 2 + bs - 1) / bs, bs>>>(
        h_edges, v_edges, h_idx, v_idx, mbrs_d, results, cells_d,
        checks.size() / 2, threshold, cells.size());
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "CUDA err: " << cudaGetErrorString(err) << std::endl;
      break;
    }
    gt.pause();
#endif
  }
  t.pause();
  return;
}
}  // namespace odrc