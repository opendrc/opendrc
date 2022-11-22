
#include <cassert>
#include <odrc/algorithm/width-check.hpp>

#include <iostream>
#include <odrc/utility/logger.hpp>
#include <odrc/utility/timer.hpp>

namespace odrc {

using odrc::core::coord;
using odrc::core::polygon;

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

__global__ void run_check(coord*        points,
                          int           num_polygons,
                          int*          offsets,
                          int           threshold,
                          check_result* vios) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= num_polygons)
    return;
  int s = offsets[tid];
  int e = offsets[tid + 1];
  for (int i = s; i < e - 1; ++i) {
    for (int j = i + 2; j < e - 1; ++j) {
      int  e11x         = points[i].x;
      int  e11y         = points[i].y;
      int  e12x         = points[i + 1].x;
      int  e12y         = points[i + 1].y;
      int  e21x         = points[j].x;
      int  e21y         = points[j].y;
      int  e22x         = points[j + 1].x;
      int  e22y         = points[j + 1].y;
      bool is_violation = false;
      // width check
      if (e11x == e12x) {  // vertical
        if (e11x < e21x) {
          // e12 e21
          // e11 e22
          bool is_inside_to_inside   = e11y < e12y and e21y > e22y;
          bool is_too_close          = e21x - e11x < threshold;
          bool is_projection_overlap = e11y < e21y and e22y < e12y;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        } else {
          // e22 e11
          // e21 e12
          bool is_inside_to_inside   = e21y < e22y and e11y > e21y;
          bool is_too_close          = e11x - e21x < threshold;
          bool is_projection_overlap = e21y < e11y and e12y < e22y;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        }
      } else {  // horizontal
        if (e11y < e22y) {
          // e21 e22
          // e12 e11
          bool is_inside_to_inside   = e11x > e12x and e21x < e22x;
          bool is_too_close          = e21y - e11y < threshold;
          bool is_projection_overlap = e21x < e11x and e12x < e22x;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        } else {
          // e11 e12
          // e22 e21
          bool is_inside_to_inside   = e21x > e22x and e11x < e12x;
          bool is_too_close          = e11y - e21y < threshold;
          bool is_projection_overlap = e11x < e21x and e22x < e12x;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        }
      }
      if (is_violation) {
        check_result& res = vios[tid];
        res.e11x          = e11x;
        res.e11y          = e11y;
        res.e12x          = e12x;
        res.e12y          = e12y;
        res.e21x          = e21x;
        res.e21y          = e21y;
        res.e22x          = e22x;
        res.e22y          = e22y;
        res.is_violation  = true;
      }
    }
  }
}

void width_check_dac23(const odrc::core::database& db,
                       int                         layer,
                       int                         threshold) {
  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  odrc::util::timer  t1("uf", logger);
  odrc::util::timer  t2("gpu", logger);
  // result memoization
  std::unordered_map<std::string, int> checked_results;

  static int    checked_poly = 0;
  static int    saved_poly   = 0;
  static int    rotated      = 0;
  static int    magnified    = 0;
  static int    reflected    = 0;
  coord*        dp;
  int*          doff;
  check_result* dv;
  cudaStream_t  stream1 = nullptr;
  cudaStreamCreate(&stream1);
  t1.start();
  cudaMallocAsync((void**)&dp, sizeof(coord) * 100000, stream1);
  cudaMallocAsync((void**)&doff, sizeof(int) * 1000, stream1);
  cudaMallocAsync((void**)&dv, sizeof(check_result) * 1000, stream1);


  std::vector<check_result> vios;

  std::vector<coord> points;
  std::vector<int>   offsets;

  for (const auto& cell : db.cells) {
    if (not cell.is_touching(layer)) {
      continue;
    }
    int local_poly = 0;
    for (const auto& polygon : cell.polygons) {
      if (polygon.layer != layer) {
        continue;
      }
      ++local_poly;
      offsets.emplace_back(points.size());
      points.insert(points.end(), polygon.points.begin(), polygon.points.end());
    }
    for (const auto& cell_ref : cell.cell_refs) {
      // should have been checked
      if (!db.get_cell(cell_ref.cell_name).is_touching(layer)) {
        continue;
      }
      auto cell_checked = checked_results.find(cell_ref.cell_name);
      assert(cell_checked != checked_results.end());
      if (cell_checked != checked_results.end() and
          !cell_ref.trans.is_magnified and !cell_ref.trans.is_reflected and
          !cell_ref.trans.is_rotated) {
        saved_poly += cell_checked->second;
      } else {
        if (cell_ref.trans.is_magnified) {
          magnified += cell_checked->second;
        }
        if (cell_ref.trans.is_reflected) {
          reflected += cell_checked->second;
        }
        if (cell_ref.trans.is_rotated) {
          rotated += cell_checked->second;
        }
      }
    }
    checked_results.emplace(cell.name, local_poly);
    checked_poly += local_poly;
  }
  offsets.emplace_back(points.size());
  t1.pause();
  t2.start();
  cudaStreamSynchronize(stream1);
  cudaMemcpy(dp, points.data(), sizeof(coord) * points.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(doff, offsets.data(), sizeof(int) * offsets.size(),
             cudaMemcpyHostToDevice);
  int np = offsets.size() - 1;
  int bs = 128;
  run_check<<<(np + bs - 1) / bs, bs>>>(dp, np, doff, threshold, dv);
  cudaDeviceSynchronize();
  //   std::cout << "checked: " << checked_poly << ", saved: " << saved_poly
  //             << "(mag: " << magnified << ", ref: " << reflected
  //             << ", rot: " << rotated << ")\n";
  t2.pause();
}
}  // namespace odrc