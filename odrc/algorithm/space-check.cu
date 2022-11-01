#include <odrc/algorithm/space-check.hpp>

#include <cassert>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <cuda_runtime.h>

namespace odrc {

using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;

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

__global__ void space_check_kernel(coord*        coords1,
                                   coord*        coords2,
                                   int           size1,
                                   int           size2,
                                   int           start,
                                   int           threshold,
                                   check_result* results) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size1 * size2 / 2) {
    return;
  }

  int p1 = tid / size2;
  int p2 = tid % size2;

  check_result& res =
      start == 0 ? results[tid] : results[tid + size1 * size2 / 4];

  res.e11x = coords1[p1].x;
  res.e11y = coords1[p1].y;
  res.e12x = coords1[p1 + 1].x;
  res.e12y = coords1[p1 + 1].y;
  res.e21x = coords2[p2].x;
  res.e21y = coords2[p2].y;
  res.e22x = coords2[p2 + 1].x;
  res.e22y = coords2[p2 + 1].y;

  // space check
  if (res.e11x == res.e12x) {  // vertical
    if (res.e11x < res.e21x) {
      // e11 e22
      // e12 e21
      bool is_outside_to_outside = res.e11y > res.e12y and res.e21y < res.e22y;
      bool is_too_close          = res.e21x - res.e11x < threshold;
      bool is_projection_overlap = res.e11y < res.e21y and res.e22y < res.e12y;
      res.is_violation =
          is_outside_to_outside and is_too_close and is_projection_overlap;
      if (res.is_violation) {
        printf("T[%d]: (%d, %d), (%d, %d), (%d, %d), (%d, %d)\n", tid, res.e11x,
               res.e11y, res.e12x, res.e12y, res.e21x, res.e21y, res.e22x,
               res.e22y);
      }

    } else {
      // e21 e12
      // e22 e11
      bool is_outside_to_outside = res.e21y > res.e22y and res.e11y < res.e21y;
      bool is_too_close          = res.e11x - res.e21x < threshold;
      bool is_projection_overlap = res.e21y < res.e11y and res.e12y < res.e22y;
      res.is_violation =
          is_outside_to_outside and is_too_close and is_projection_overlap;
    }
  } else {  // horizontal
    if (res.e11y < res.e22y) {
      // e22 e21
      // e11 e12
      bool is_outside_to_outside = res.e11x < res.e12x and res.e21x > res.e22x;
      bool is_too_close          = res.e21y - res.e11y < threshold;
      bool is_projection_overlap = res.e21x < res.e11x and res.e12x < res.e22x;
      res.is_violation =
          is_outside_to_outside and is_too_close and is_projection_overlap;
    } else {
      // e12 e11
      // e21 e22
      bool is_outside_to_outside = res.e21x < res.e22x and res.e11x > res.e12x;
      bool is_too_close          = res.e11y - res.e21y < threshold;
      bool is_projection_overlap = res.e11x < res.e21x and res.e22x < res.e12x;
      res.is_violation =
          is_outside_to_outside and is_too_close and is_projection_overlap;
    }
  }
}

void run_space_check(const polygon&   polygon1,
                     const polygon&   polygon2,
                     int              threshold,
                     coord*           coords1,
                     coord*           coords2,
                     check_result*    results,
                     check_result*    result_host,
                     cudaGraphExec_t& graph_exec) {
  cudaStream_t stream1 = nullptr;
  cudaStream_t stream2 = nullptr;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  int         size1 = (polygon1.points.size() - 1) / 2;
  int         size2 = (polygon2.points.size() - 1) / 2;
  cudaGraph_t graph;
  // Two streams will be launched:
  //   s
  //  / \  // to avoid -Wcomment due to '\'
  // s1 s2
  //  \ /
  //   e
  cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
  // construct cuda graph for width check here
  cudaMemcpyAsync(coords1, polygon1.points.data(),
                  sizeof(coord) * polygon1.points.size(),
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(coords2, polygon2.points.data(),
                  sizeof(coord) * polygon2.points.size(),
                  cudaMemcpyHostToDevice, stream2);
  cudaEvent_t e1;  // sync memcpy
  cudaEventCreate(&e1);
  cudaEventRecord(e1, stream1);
  cudaStreamWaitEvent(stream2, e1);

  int num_parallel_checks = size1 * size2 / 4;
  space_check_kernel<<<(num_parallel_checks + 127) / 128, 128, 0, stream1>>>(
      coords1, coords2, size1, size2, 0, threshold, results);
  space_check_kernel<<<(num_parallel_checks + 127) / 128, 128, 0, stream2>>>(
      coords1, coords2, size1, size2, 1, threshold, results);

  cudaEvent_t e2;  // sync all kernel launch
  cudaEventCreate(&e2);
  cudaEventRecord(e2, stream2);
  cudaStreamWaitEvent(stream1, e2);
  cudaMemcpyAsync(result_host, results,
                  sizeof(check_result) * num_parallel_checks * 2,
                  cudaMemcpyDeviceToHost, stream1);
  cudaStreamEndCapture(stream1, &graph);
  auto perror = cudaGetLastError();
  if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
    throw std::runtime_error("CUDA pre error: " + std::to_string(perror));
  }

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
  if (graph_exec == nullptr || update_result != cudaGraphExecUpdateSuccess) {
    // If a previous update failed, destroy the cudaGraphExec_t
    // before re-instantiating it
    if (graph_exec != nullptr) {
      cudaGraphExecDestroy(graph_exec);
    }
    // Instantiate graphExec from graph. The error node and
    // error message parameters are unused here.
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
  }
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaGraphDestroy(graph);
  cudaGraphLaunch(graph_exec, 0);
  cudaDeviceSynchronize();
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {  // TODO: change to OpenDRC exception
    throw std::runtime_error("CUDA error: " + std::to_string(error));
  }
}

void space_check(const odrc::core::database& db,
                 const int                   layer1,
                 const int                   layer2,
                 const int                   threshold) {
  cudaStream_t    stream;
  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t     error;
  coord*          coord_buffer1      = nullptr;
  coord*          coord_buffer2      = nullptr;
  check_result*   check_results      = nullptr;
  check_result*   check_results_host = nullptr;

  // TODO: remove magic numbers
  cudaStreamCreate(&stream);
  cudaMallocAsync((void**)&coord_buffer1, sizeof(coord) * 201, stream);
  cudaMallocAsync((void**)&coord_buffer2, sizeof(coord) * 201, stream);
  cudaMallocAsync((void**)&check_results, sizeof(check_result) * 20000, stream);
  cudaMallocHost((void**)&check_results_host, sizeof(check_result) * 20000);
  error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {  // TODO: change to OpenDRC exception
    throw std::runtime_error("CUDA error: " + std::to_string(error));
  }
  cudaStreamDestroy(stream);

  // result memoiozation
  std::unordered_map<std::string, bool> checked_results;

  // The structure to represent an inter-poly check.
  // It cares about layer1 for cell1 and layer2 for cell2.
  // Only the top-most cell is not cell_ref.
  struct _task {
    std::variant<const polygon*, const cell_ref*> object1;
    const cell_ref*                               object2;
    bool                                          is_every_subtask_done = false;
  };

  cell_ref top_cell_ref{db.cells.back().name, odrc::core::coord{0, 0}, {}};

  std::stack<_task> tasks;
  tasks.push(_task{&top_cell_ref, &top_cell_ref, false});

  while (not tasks.empty()) {
    auto task = tasks.top();
    tasks.pop();
    const cell_ref** r1    = std::get_if<const cell_ref*>(&task.object1);
    const auto&      cell2 = db.get_cell(task.object2->cell_name);
    if (task.is_every_subtask_done) {  // just to see if memoization is possible
      if (r1 != nullptr and *r1 == task.object2) {
        checked_results.emplace((*r1)->cell_name, true);
      }
      continue;
    }
    if (r1 == nullptr) {  // object1 is polygon
      tasks.push(_task{task.object1, task.object2, true});
      const auto& polygon1 = *std::get<const polygon*>(task.object1);
      assert(polygon1.layer == layer1);  // should be enqueued otherwise

      // polygon vs polygon are sent to run_space_check directly

      for (const auto& polygon2 : cell2.polygons) {
        if (polygon2.layer == layer2 and polygon1.is_touching(polygon2)) {
          run_space_check(polygon1, polygon2, threshold, coord_buffer1,
                          coord_buffer2, check_results, check_results_host,
                          graph_exec);
        }
      }

      // polygon vs cell_ref are enqueued

      for (const auto& cell_ref2 : cell2.cell_refs) {
        const auto& the_cell = db.get_cell(cell_ref2.cell_name);
        if (the_cell.is_touching(layer2) and cell_ref2.is_touching(polygon1)) {
          tasks.push(_task{&polygon1, &cell_ref2, false});
        }
      }

    } else {  // object1 is cell

      // NOTE: (a^M b^M) and (b^M a^M) duplicates
      // to fix, assign arbitrary id to cells to mark unique checks
      const auto& cell1 =
          db.get_cell(std::get<const cell_ref*>(task.object1)->cell_name);
      tasks.push(_task{task.object1, task.object2, true});
      for (const auto& polygon1 : cell1.polygons) {
        if (polygon1.layer != layer1) {
          continue;
        }
        // polygon vs polygon are sent to run_space_check directly
        for (const auto& polygon2 : cell2.polygons) {
          if (polygon2.layer == layer2 and polygon1.is_touching(polygon2)) {
            run_space_check(polygon1, polygon2, threshold, coord_buffer1,
                            coord_buffer2, check_results, check_results_host,
                            graph_exec);
          }
        }
        // polygon vs cell_ref are enqueued
        for (const auto& cell_ref2 : cell2.cell_refs) {
          const auto& the_cell = db.get_cell(cell_ref2.cell_name);
          if (the_cell.is_touching(layer2) and
              cell_ref2.is_touching(polygon1)) {
            tasks.push(_task{&polygon1, &cell_ref2, false});
          }
        }
      }

      // cell_ref vs cell_ref are enqueued
      for (const auto& cell_ref1 : cell1.cell_refs) {
        const auto& the_cell1 = db.get_cell(cell_ref1.cell_name);
        if (the_cell1.is_touching(layer1)) {
          for (const auto& cell_ref2 : cell2.cell_refs) {
            const auto& the_cell2 = db.get_cell(cell_ref2.cell_name);
            if (the_cell2.is_touching(layer2) and
                cell_ref1.is_touching(cell_ref2)) {
              tasks.push(_task{&cell_ref1, &cell_ref2, false});
            }
          }
        }
      }
    }
  }
}
}  // namespace odrc