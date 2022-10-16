#include <odrc/algorithm/width-check.hpp>

#include <algorithm>
#include <utility>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/pair.h>

#include <odrc/infrastructure/execution.hpp>
#include <odrc/infrastructure/sweepline/sweepline.hpp>

#include <odrc/interface/gdsii/gdsii.hpp>

namespace odrc {
using horizontal_edge = thrust::pair<thrust::pair<int, int>, int>;
using vertical_edge   = thrust::pair<int, thrust::pair<int, int>>;

struct check_result {
  int  e1           = -1;
  int  e2           = -1;
  bool is_violation = false;
};

__global__ void _horizontal_width_check_kernel(horizontal_edge* edges,
                                               int*             prefix,
                                               check_result*    buffer,
                                               int              total_edges) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < total_edges) {
    int start_edge = prefix[i];
    int vio_count  = 0;
    for (int e = start_edge; e < i; ++e) {
      if (edges[e].first.second < edges[i].first.first or
          edges[e].first.first > edges[i].first.second) {
        continue;  // no violation
      } else {
        buffer[i * 10 + vio_count] = check_result{e, i, true};
        ++vio_count;
        if (vio_count >= 10)
          return;
      }
    }
  }
}

__global__ void _vertical_width_check_kernel(vertical_edge* edges,
                                             int*           prefix,
                                             check_result*  buffer,
                                             int            total_edges) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < total_edges) {
    int start_edge = prefix[i];
    int vio_count  = 0;
    for (int e = start_edge; e < i; ++e) {
      if (edges[e].second.second < edges[i].second.first or
          edges[e].second.first > edges[i].second.second) {
        continue;  // no violation
      } else {
        buffer[i * 10 + vio_count] = check_result{e, i, true};
        ++vio_count;
        if (vio_count >= 10)
          return;
      }
    }
  }
}

void width_check(const odrc::gdsii::library& lib, int layer, int threshold) {
  for (auto&& s : lib.structs) {
    for (auto&& e : s.elements) {
      if (e->rtype == odrc::gdsii::record_type::BOUNDARY){
        gdsii::library::boundary* ptrtmp = static_cast<gdsii::library::boundary*>(e);
        if(ptrtmp->layer == layer) {
        std::vector<horizontal_edge> horizontal_edges;
        std::vector<vertical_edge>   vertical_edges;
        int                          num_points = ptrtmp->points.size() - 1;
        for (int i = 0; i < num_points - 1; ++i) {
          if (ptrtmp->points[i].x == ptrtmp->points[i + 1].x) {  // vertical
            vertical_edges.emplace_back(std::pair{
                ptrtmp->points[i].x, std::pair{ptrtmp->points[i].y, ptrtmp->points[i + 1].y}});
          } else {
            horizontal_edges.emplace_back(std::pair{
                std::pair{ptrtmp->points[i].x, ptrtmp->points[i + 1].x}, ptrtmp->points[i].y});
          }
        }
        std::sort(
            horizontal_edges.begin(), horizontal_edges.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        std::sort(
            vertical_edges.begin(), vertical_edges.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });


        thrust::device_vector<horizontal_edge> horizontal_edges_dev(
            horizontal_edges);
        int* horizontal_prefix_dev;
        cudaMalloc(reinterpret_cast<void**>(&horizontal_prefix_dev),
                   sizeof(int) * horizontal_edges_dev.size());
        odrc::core::sweepline(
            odrc::execution::par,
            thrust::raw_pointer_cast(horizontal_edges_dev.data()),
            thrust::raw_pointer_cast(horizontal_edges_dev.data() +
                                     horizontal_edges.size()),
            horizontal_prefix_dev,
            [threshold] __device__(auto s, auto e) -> int {
              for (auto it = e - 2; it >= s; --it) {
                if ((e - 1)->second - it->second > threshold) {
                  return std::distance(s, it + 1);
                }
              }
              return 0;
            });

        check_result* result_buffer;
        cudaMalloc(reinterpret_cast<void**>(&result_buffer),
                   sizeof(check_result) * 10 * horizontal_edges.size());
        _horizontal_width_check_kernel<<<(horizontal_edges.size() + 127) / 128,
                                         128>>>(
            thrust::raw_pointer_cast(horizontal_edges_dev.data()),
            horizontal_prefix_dev, result_buffer, horizontal_edges.size());


        thrust::device_vector<vertical_edge> vertical_edges_dev(vertical_edges);
        int*                                 vertical_prefix_dev;
        cudaMalloc(reinterpret_cast<void**>(&vertical_prefix_dev),
                   sizeof(int) * vertical_edges_dev.size());
        odrc::core::sweepline(
            odrc::execution::par,
            thrust::raw_pointer_cast(vertical_edges_dev.data()),
            thrust::raw_pointer_cast(vertical_edges_dev.data() +
                                     vertical_edges.size()),
            vertical_prefix_dev, [threshold] __device__(auto s, auto e) -> int {
              for (auto it = e - 2; it >= s; --it) {
                if ((e - 1)->first - it->first > threshold) {
                  return std::distance(s, it + 1);
                }
              }
              return 0;
            });
        cudaFree(result_buffer);
        cudaMalloc(reinterpret_cast<void**>(&result_buffer),
                   sizeof(check_result) * 10 * vertical_edges.size());
        _vertical_width_check_kernel<<<(vertical_edges.size() + 127) / 128,
                                       128>>>(
            thrust::raw_pointer_cast(vertical_edges_dev.data()),
            vertical_prefix_dev, result_buffer, vertical_edges.size());
      }
    }
    }
  }
}
}  // namespace odrc