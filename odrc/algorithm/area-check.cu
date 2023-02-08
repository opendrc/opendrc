
#include <odrc/algorithm/parallel_mode.hpp>

#include <iostream>
#include <odrc/core/cell.hpp>
#include <odrc/core/rule.hpp>
#include <unordered_map>
#include <vector>

namespace odrc {

using odrc::core::polygon;

__global__ void run_check(coord* points,
                          int    num_polygons,
                          int*   offsets,
                          int    threshold,
                          int**  violations) {
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
  if (area < threshold) {
    violations[tid] = &s;
  }
}

void area_check_par(odrc::core::database&         db,
                    int                           layer,
                    int                           threshold,
                    std::vector<core::violation>& vios) {
  // result memoization
  std::unordered_map<std::string, std::pair<int, int>> checked_results;

  static int   checked_poly = 0;
  static int   saved_poly   = 0;
  static int   rotated      = 0;
  static int   magnified    = 0;
  static int   reflected    = 0;
  coord*       dp;
  int*         doff;
  int**        dv;
  cudaStream_t stream1 = nullptr;
  cudaStreamCreate(&stream1);
  cudaMallocAsync((void**)&dp, sizeof(coord) * 100000, stream1);
  cudaMallocAsync((void**)&doff, sizeof(int) * 1000, stream1);
  cudaMallocAsync((void**)&dv, sizeof(int*) * 1000, stream1);

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
    checked_results.emplace(
        cell.name, std::make_pair(checked_poly, local_poly + checked_poly));
    checked_poly += local_poly;
  }
  offsets.emplace_back(points.size());
  cudaStreamSynchronize(stream1);
  cudaMemcpy(dp, points.data(), sizeof(coord) * points.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(doff, offsets.data(), sizeof(int) * offsets.size(),
             cudaMemcpyHostToDevice);
  int np = offsets.size() - 1;
  int bs = 128;
  run_check<<<(np + bs - 1) / bs, bs>>>(dp, np, doff, threshold, dv);
  cudaDeviceSynchronize();
}
}  // namespace odrc