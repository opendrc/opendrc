#include <cassert>
#include <odrc/algorithm/width-check.hpp>

#include <iostream>

namespace odrc {

using odrc::core::coord;
using odrc::core::polygon;

__global__ void xcheck_kernel(const coord* points,
                              int*         offsets,
                              int          size,
                              int          threshold) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size) {
    return;
  }
  int start = offsets[tid];
  int end   = offsets[tid + 1];
  for (int i = start; i < end - 1; ++i) {
    for (int j = i + 2; j < end - 1; ++j) {
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
    }
  }
}

void width_xcheck(const odrc::core::database& db, int layer, int threshold) {
  std::cout << "### width_xcheck ###" << std::endl;
  coord*             points_dev;
  int*               offset_dev;
  std::vector<coord> points_host;
  std::vector<int>   offset_host;

  //   for (const auto& cell : db.cells) {
  const auto& cell = db.cells.back();
  //   if (not cell.is_touching(layer)) {
  //     continue;
  //   }
  std::cout << "#raw polygons: " << cell.polygons.size() << std::endl;
  for (const auto& polygon : cell.polygons) {
    if (polygon.layer != layer) {
      continue;
    }
    offset_host.emplace_back(points_host.size());
    points_host.insert(points_host.end(), polygon.points.begin(),
                       polygon.points.end());
  }
  std::cout << "#cell refs: " << cell.cell_refs.size() << std::endl;
  for (const auto& cell_ref : cell.cell_refs) {
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (not the_cell.is_touching(layer)) {
      continue;
    }
    for (const auto& polygon : the_cell.polygons) {
      if (polygon.layer != layer) {
        continue;
      }
      offset_host.emplace_back(points_host.size());
      points_host.insert(points_host.end(), polygon.points.begin(),
                         polygon.points.end());
    }
  }
  //   }
  std::cout << "#polygons: " << offset_host.size() << std::endl;
  std::cout << "#edges: " << points_host.size() << std::endl;
  // sentinel to mark the end
  offset_host.emplace_back(points_host.size());

  cudaMalloc((void**)&points_dev, sizeof(coord) * points_host.size());
  cudaMalloc((void**)&offset_dev, sizeof(int) * offset_host.size());
  cudaMemcpy(points_dev, points_host.data(), sizeof(coord) * points_host.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(offset_dev, offset_host.data(), sizeof(coord) * offset_host.size(),
             cudaMemcpyHostToDevice);
  xcheck_kernel<<<(offset_host.size() - 1 + 127) / 128, 128>>>(
      points_dev, offset_dev, offset_host.size() - 1, threshold);
  cudaDeviceSynchronize();
}

}  // namespace odrc