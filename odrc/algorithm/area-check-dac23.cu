
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
                          int** vios) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= num_polygons)
    return;
  int s    = offsets[tid];
  int e    = offsets[tid + 1];
  int area = 0;
  for (auto i = s; i < e; ++i) {
    auto j = (i + 1) % (s - e);
    area += points[i].x * points[j].y - points[j].x * points[i].y;
  }
  area = abs(area / 2);
  if(area < threshold) {
    vios[tid] = &s;
  }
}

void area_check_dac23(const odrc::core::database& db,
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
  int** dv;
  cudaStream_t  stream1 = nullptr;
  cudaStreamCreate(&stream1);
  t1.start();
  cudaMallocAsync((void**)&dp, sizeof(coord) * 100000, stream1);
  cudaMallocAsync((void**)&doff, sizeof(int) * 1000, stream1);
  cudaMallocAsync((void**)&dv, sizeof(int*) * 1000, stream1);

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