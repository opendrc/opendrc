#include <odrc/algorithm/width-check.hpp>

#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>

namespace odrc {
using coord = odrc::core::coord;

__global__ void check_kernel(coord*        coords,
                             int           start,
                             int           size,
                             int           threshold,
                             check_result* results) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= (size - 1) * size / 2) {
    return;
  }

  int offset       = start;
  int num_checks   = size - 1;
  int total_checks = 0;

  check_result& res =
      start == 0 ? results[tid] : results[tid + (size - 1) * size / 2];

  // calculate offset
  for (int i = 0; i < size; ++i) {
    if (tid - total_checks < num_checks)
      break;

    total_checks += num_checks;
    offset += 2;
    num_checks -= 2;
  }
  int r = offset + (tid - total_checks + 1) * 2;

  res.e11x = coords[offset].x;
  res.e11y = coords[offset].y;
  res.e12x = coords[offset + 1].x;
  res.e12y = coords[offset + 1].y;
  res.e21x = coords[r].x;
  res.e21y = coords[r].y;
  res.e22x = coords[r + 1].x;
  res.e22y = coords[r + 1].y;

  // width check
  if (res.e11x == res.e12x) {  // vertical
    if (res.e11x < res.e21x) {
      // e12 e21
      // e11 e22
      bool is_inside_to_inside   = res.e11y < res.e12y and res.e21y > res.e22y;
      bool is_too_close          = res.e21x - res.e11x < threshold;
      bool is_projection_overlap = res.e11y < res.e21y and res.e22y < res.e12y;
      res.is_violation =
          is_inside_to_inside and is_too_close and is_projection_overlap;
      if (res.is_violation) {
        printf("T[%d]: (%d, %d), (%d, %d), (%d, %d), (%d, %d)\n", tid, res.e11x,
               res.e11y, res.e12x, res.e12y, res.e21x, res.e21y, res.e22x,
               res.e22y);
      }
    } else {
      // e22 e11
      // e21 e12
      bool is_inside_to_inside   = res.e21y < res.e22y and res.e11y > res.e21y;
      bool is_too_close          = res.e11x - res.e21x < threshold;
      bool is_projection_overlap = res.e21y < res.e11y and res.e12y < res.e22y;
      res.is_violation =
          is_inside_to_inside and is_too_close and is_projection_overlap;
    }
  } else {  // horizontal
    if (res.e11y < res.e22y) {
      // e21 e22
      // e12 e11
      bool is_inside_to_inside   = res.e11x > res.e12x and res.e21x < res.e22x;
      bool is_too_close          = res.e21y - res.e11y < threshold;
      bool is_projection_overlap = res.e21x < res.e11x and res.e12x < res.e22x;
      res.is_violation =
          is_inside_to_inside and is_too_close and is_projection_overlap;
    } else {
      // e11 e12
      // e22 e21
      bool is_inside_to_inside   = res.e21x > res.e22x and res.e11x < res.e12x;
      bool is_too_close          = res.e11y - res.e21y < threshold;
      bool is_projection_overlap = res.e11x < res.e21x and res.e22x < res.e12x;
      res.is_violation =
          is_inside_to_inside and is_too_close and is_projection_overlap;
    }
  }
}

void width_check_pal(const odrc::core::database& db,
                     int                         layer,
                     int                         threshold,
                     std::vector<check_result>&  vios) {
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  coord*        coord_buffer       = nullptr;
  check_result* check_results      = nullptr;
  check_result* check_results_host = nullptr;

  // TODO: remove magic numbers
  cudaError_t error;
  cudaMallocAsync((void**)&coord_buffer, sizeof(coord) * 201, stream1);
  cudaMallocAsync((void**)&check_results, sizeof(check_result) * 9900, stream1);
  cudaMallocHost((void**)&check_results_host, sizeof(check_result) * 9900);
  error = cudaStreamSynchronize(stream1);

  // result memoization
  std::unordered_map<std::string, int> checked_results;

  static int checked_poly = 0;
  static int saved_poly   = 0;

  cudaGraph_t     graph      = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
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
      // Two streams will be launched:
      //   s
      //  / \  // to avoid -Wcomment due to '\'
      // s1 s2
      //  \ /
      //   e
      using coord = odrc::core::coord;
      cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
      // construct cuda graph for width check here
      cudaMemcpyAsync(coord_buffer, polygon.points.data(),
                      sizeof(coord) * polygon.points.size(),
                      cudaMemcpyHostToDevice, stream1);
      cudaEvent_t e1;  // sync memcpy
      cudaEventCreate(&e1);
      cudaEventRecord(e1, stream1);
      cudaStreamWaitEvent(stream2, e1);

      int num_parallel_edges = (polygon.points.size() - 1) / 2;
      int num_checks = (num_parallel_edges - 1) * num_parallel_edges / 2;

      check_kernel<<<(num_checks + 127) / 128, 128, 0, stream1>>>(
          coord_buffer, 0, num_parallel_edges, threshold, check_results);
      check_kernel<<<(num_checks + 127) / 128, 128, 0, stream2>>>(
          coord_buffer, 1, num_parallel_edges, threshold, check_results);

      cudaEvent_t e2;  // sync all kernel launch
      cudaEventCreate(&e2);
      cudaEventRecord(e2, stream2);
      cudaStreamWaitEvent(stream1, e2);
      cudaMemcpyAsync(check_results_host, check_results,
                      sizeof(check_result) * num_checks * 2,
                      cudaMemcpyDeviceToHost, stream1);
      cudaStreamEndCapture(stream1, &graph);

      cudaGraphExecUpdateResult update_result;
      // If we've already instantiated the graph, try to update it directly
      // and avoid the instantiation overhead
      if (graph_exec != nullptr) {
        cudaGraphNode_t error_node;
        // If the graph fails to update, errorNode will be set to the
        // node causing the failure and updateResult will be set to a
        // reason code.
        cudaGraphExecUpdate(graph_exec, graph, &error_node, &update_result);
      }

      // Instantiate during the first iteration or whenever the update
      // fails for any reason
      if (graph_exec == nullptr ||
          update_result != cudaGraphExecUpdateSuccess) {
        // If a previous update failed, destroy the cudaGraphExec_t
        // before re-instantiating it
        if (graph_exec != nullptr) {
          cudaGraphExecDestroy(graph_exec);
        }
        // Instantiate graphExec from graph. The error node and
        // error message parameters are unused here.
        cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
      }
      cudaGraphDestroy(graph);
      cudaGraphLaunch(graph_exec, 0);
      cudaDeviceSynchronize();
      error = cudaGetLastError();
      if (error != 0) {  // TODO: change to OpenDRC exception
        throw std::runtime_error("CUDA error: " + std::to_string(error));
      }
    }
    for (const auto& cell_ref : cell.cell_refs) {
      // should have been checked
      auto cell_checked = checked_results.find(cell_ref.cell_name);
      if (cell_checked != checked_results.end()) {
        saved_poly += cell_checked->second;
      }
    }
    checked_results.emplace(cell.name, local_poly);
    checked_poly += local_poly;
  }
}
}  // namespace odrc