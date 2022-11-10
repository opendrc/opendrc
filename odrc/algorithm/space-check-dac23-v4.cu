#include <odrc/algorithm/space-check.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include <odrc/core/interval_tree.hpp>
#include <odrc/utility/logger.hpp>
#include <odrc/utility/timer.hpp>

namespace odrc {

using coord    = odrc::core::coord;
using polygon  = odrc::core::polygon;
using cell_ref = odrc::core::cell_ref;
using h_edge   = odrc::core::h_edge;
using v_edge   = odrc::core::v_edge;

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

__global__ void check_vertical(v_edge*       v1,
                               v_edge*       v2,
                               int           size1,
                               int           size2,
                               int           threshold,
                               check_result* results) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size1 * size2) {
    return;
  }
  int p1 = tid / size2;
  int p2 = tid % size2;

  check_result& res = results[tid];

  res.e11x = v1[p1].x;
  res.e11y = v1[p1].y1;
  res.e12x = v1[p1].x;
  res.e12y = v1[p1].y2;
  res.e21x = v2[p2].x;
  res.e21y = v2[p2].y1;
  res.e22x = v2[p2].x;
  res.e22y = v2[p2].y2;

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
}

__global__ void check_horizontal(h_edge*       h1,
                                 h_edge*       h2,
                                 int           size1,
                                 int           size2,
                                 int           threshold,
                                 check_result* results) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= size1 * size2) {
    return;
  }
  int p1 = tid / size2;
  int p2 = tid % size2;

  check_result& res = results[tid];

  res.e11x = h1[p1].x1;
  res.e11y = h1[p1].y;
  res.e12x = h1[p1].x2;
  res.e12y = h1[p1].y;
  res.e21x = h2[p2].x1;
  res.e21y = h2[p2].y;
  res.e22x = h2[p2].x2;
  res.e22y = h2[p2].y;

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

void run_space_check(const h_edge*    h1,
                     int              sizeh1,
                     const h_edge*    h2,
                     int              sizeh2,
                     const v_edge*    v1,
                     int              sizev1,
                     const v_edge*    v2,
                     int              sizev2,
                     h_edge*          buf_h1,
                     h_edge*          buf_h2,
                     v_edge*          buf_v1,
                     v_edge*          buf_v2,
                     check_result*    res_h,
                     check_result*    res_v,
                     check_result*    res_h_host,
                     check_result*    res_v_host,
                     int              threshold,
                     cudaGraphExec_t& graph_exec) {
  // std::cout << sizeh1 << " " << sizeh2 << " " << sizev1 << " " << sizev2
  //           << std::endl;
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStream_t stream3;
  cudaStream_t stream4;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  cudaStreamCreate(&stream4);
  assert(sizeh1 < 1000);
  assert(sizeh2 < 1000);
  assert(sizev1 < 1000);
  assert(sizev2 < 1000);
  cudaGraph_t graph;
  // Two streams will be launched:
  //   s
  //  / \  // to avoid -Wcomment due to '\'
  // s1 s2
  //  \ /
  //   e
  cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
  cudaEvent_t e0;
  cudaEventCreate(&e0);
  cudaEventRecord(e0, stream1);
  cudaStreamWaitEvent(stream2, e0);
  cudaStreamWaitEvent(stream3, e0);
  cudaStreamWaitEvent(stream4, e0);

  if (sizeh1 > 0 and sizeh2 > 0) {
    cudaMemcpyAsync(buf_h1, h1, sizeof(h_edge) * sizeh1, cudaMemcpyHostToDevice,
                    stream1);
    cudaMemcpyAsync(buf_h2, h2, sizeof(h_edge) * sizeh2, cudaMemcpyHostToDevice,
                    stream2);
    auto perror = cudaGetLastError();
    if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
      throw std::runtime_error("CUDA error1: " + std::to_string(perror));
    }
    cudaEvent_t e1;  // sync memcpy for h
    cudaEventCreate(&e1);
    cudaEventRecord(e1, stream2);
    cudaStreamWaitEvent(stream1, e1);
    perror = cudaGetLastError();
    if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
      throw std::runtime_error("CUDA error2: " + std::to_string(perror));
    }

    int hnum_parallel_checks = sizeh1 * sizeh2;
    check_horizontal<<<(hnum_parallel_checks + 127) / 128, 128, 0, stream1>>>(
        buf_h1, buf_h2, sizeh1, sizeh2, threshold, res_h);
    perror = cudaGetLastError();
    if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
      throw std::runtime_error("CUDA error3: " + std::to_string(perror));
    }
    cudaMemcpyAsync(res_h_host, res_h,
                    sizeof(check_result) * hnum_parallel_checks,
                    cudaMemcpyDeviceToHost, stream1);
    perror = cudaGetLastError();
    if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
      throw std::runtime_error("CUDA error4: " + std::to_string(perror));
    }
  }
  if (sizev1 > 0 and sizev2 > 0) {
    cudaMemcpyAsync(buf_v1, v1, sizeof(v_edge) * sizev1, cudaMemcpyHostToDevice,
                    stream3);
    cudaMemcpyAsync(buf_v2, v2, sizeof(v_edge) * sizev2, cudaMemcpyHostToDevice,
                    stream4);
    auto perror = cudaGetLastError();
    if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
      throw std::runtime_error("CUDA error4.5: " + std::to_string(perror));
    }
    cudaEvent_t e2;  // sync memcpy for v
    cudaEventCreate(&e2);
    perror = cudaEventRecord(e2, stream4);
    if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
      throw std::runtime_error("CUDA error5: " + std::to_string(perror));
    }
    perror = cudaStreamWaitEvent(stream3, e2);
    if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
      throw std::runtime_error("CUDA error6: " + std::to_string(perror));
    }
    int vnum_parallel_checks = sizev1 * sizev2;
    check_vertical<<<(vnum_parallel_checks + 127) / 128, 128, 0, stream3>>>(
        buf_v1, buf_v2, sizev1, sizev2, threshold, res_v);
    cudaMemcpyAsync(res_v_host, res_v,
                    sizeof(check_result) * vnum_parallel_checks,
                    cudaMemcpyDeviceToHost, stream3);
  }

  cudaEvent_t e3;  // sync all kernel launch and copy back
  cudaEventCreate(&e3);
  cudaEventRecord(e3, stream3);
  auto perror = cudaStreamWaitEvent(stream1, e3);
  if (perror != cudaSuccess) {  // TODO: change to OpenDRC exception
    throw std::runtime_error("CUDA error888: " + std::to_string(perror));
  }
  cudaStreamEndCapture(stream1, &graph);
  perror = cudaGetLastError();
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
    cudaGetLastError();  // clear error message
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
  cudaStreamDestroy(stream3);
  cudaStreamDestroy(stream4);
  cudaGraphDestroy(graph);
  cudaGraphLaunch(graph_exec, 0);
}

void space_check_dac23(const odrc::core::database& db,
                       int                         layer1,
                       int                         layer2,
                       int                         threshold) {
  cudaStream_t    stream;
  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t     error;
  h_edge*         coord_bufferh1      = nullptr;
  h_edge*         coord_bufferh2      = nullptr;
  v_edge*         coord_bufferv1      = nullptr;
  v_edge*         coord_bufferv2      = nullptr;
  check_result*   check_resultsh      = nullptr;
  check_result*   check_resultsv      = nullptr;
  check_result*   check_resultsh_host = nullptr;
  check_result*   check_resultsv_host = nullptr;

  // TODO: remove magic numbers
  cudaStreamCreate(&stream);
  cudaMallocAsync((void**)&coord_bufferh1, sizeof(h_edge) * 1000, stream);
  cudaMallocAsync((void**)&coord_bufferh2, sizeof(h_edge) * 1000, stream);
  cudaMallocAsync((void**)&coord_bufferv1, sizeof(v_edge) * 1000, stream);
  cudaMallocAsync((void**)&coord_bufferv2, sizeof(v_edge) * 1000, stream);
  cudaMallocAsync((void**)&check_resultsh, sizeof(check_result) * 1000000,
                  stream);
  cudaMallocAsync((void**)&check_resultsv, sizeof(check_result) * 1000000,
                  stream);
  cudaMallocHost((void**)&check_resultsh_host, sizeof(check_result) * 1000000);
  cudaMallocHost((void**)&check_resultsv_host, sizeof(check_result) * 1000000);
  error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) {  // TODO: change to OpenDRC exception
    throw std::runtime_error("CUDA error: " + std::to_string(error));
  }
  cudaStreamDestroy(stream);

  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  using Intvl = core::interval<int, int>;
  struct event {
    Intvl intvl;
    int   y;
    int   offset;
    bool  is_polygon;
    bool  is_inevent;
  };

  std::vector<event> events;

  const auto& top_cell = db.cells.back();

  std::cout << "# polygons: " << top_cell.polygons.size() << std::endl;

  // for (int i = 0; i < top_cell.polygons.size(); ++i) {
  //   const auto& polygon = top_cell.polygons.at(i);
  //   if (polygon.layer != layer1 and polygon.layer != layer2) {
  //     continue;
  //   }
  //   events.emplace_back(
  //       event{core::interval{polygon.mbr[0], polygon.mbr[1], polygon.mbr[2],
  //       i},
  //             i, true, true});
  //   events.emplace_back(
  //       event{core::interval{polygon.mbr[0], polygon.mbr[1], polygon.mbr[3],
  //       i},
  //             i, true, false});
  // }
  std::cout << "# cell_refs: " << top_cell.cell_refs.size() << std::endl;
  for (int i = 0; i < top_cell.cell_refs.size(); ++i) {
    const auto& cell_ref = top_cell.cell_refs.at(i);
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (!the_cell.is_touching(layer1) and !the_cell.is_touching(layer2)) {
      continue;
    }
    events.emplace_back(
        event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3],
                    i + int(db.cells.back().polygons.size()) * 0},
              cell_ref.mbr[0], i, false, true});
    events.emplace_back(
        event{Intvl{cell_ref.mbr[2], cell_ref.mbr[3],
                    (i + int(db.cells.back().polygons.size()) * 0)},
              cell_ref.mbr[1], i, false, false});
  }
  std::cout << "# events: " << events.size() << std::endl;

  {
    odrc::util::timer t("t", logger);
    t.start();

    std::sort(events.begin(), events.end(), [](const auto& e1, const auto& e2) {
      if (e1.y == e2.y) {
        return e1.is_inevent and !e2.is_inevent;
      } else {
        return e1.y < e2.y;
      }
    });
    t.pause();
  }

  core::interval_tree<int, int> tree;

  std::cout << "scanning bottom up" << std::endl;
  int                                   add = 0;
  int                                   del = 0;
  int                                   cnt = 0;
  int                                   sum = 0;
  std::unordered_map<int, const event*> map;
  int                                   total_ovlp = 0;
  {
    odrc::util::timer t("t", logger);
    t.start();
    // hack to balance tree
    // Intvl _i1{29999, 30001, -1};
    // Intvl _i2{14999, 15001, -1};
    // Intvl _i3{44999, 45001, -1};
    // tree.insert(_i1);
    // tree.insert(_i2);
    // tree.insert(_i3);
    // tree.remove(_i1);
    // tree.remove(_i2);
    // tree.remove(_i3);
    int next_progress = 5;
    for (int i = 0; i < events.size(); ++i) {
      if (double(i) / events.size() >= next_progress / 100.0) {
        std::cout << "Progress : " << next_progress << "%" << std::endl;
        next_progress += 5;
      }
      //   for (const auto& e : events) {
      const auto& e = events[i];
      if (e.is_inevent) {
        auto ovlp = tree.get_intervals_overlapping_with(e.intvl);
        for (int c : ovlp) {
          cudaDeviceSynchronize();
          auto error = cudaGetLastError();
          if (error != cudaSuccess) {  // TODO: change to OpenDRC exception
            throw std::runtime_error("CUDA error: " + std::to_string(error));
          }
          const auto& cr1 = top_cell.cell_refs.at(c);
          const auto& cr2 = top_cell.cell_refs.at(e.intvl.v);
          if (cr1.h_edges.empty() or cr2.h_edges.empty() or
              cr1.v_edges.empty() or cr2.v_edges.empty()) {
            continue;
          }
          int hmin1 = cr1.h_edges.front().y;
          int hmax1 = cr1.h_edges.back().y;
          int hmin2 = cr2.h_edges.front().y;
          int hmax2 = cr2.h_edges.back().y;
          int vmin1 = cr1.v_edges.front().x;
          int vmax1 = cr1.v_edges.back().x;
          int vmin2 = cr2.v_edges.front().x;
          int vmax2 = cr2.v_edges.back().x;
          // std::cout << hmin1 << " to " << hmax1 << std::endl;
          // std::cout << hmin2 << " to " << hmax2 << std::endl;
          // std::cout << vmin1 << " to " << vmax1 << std::endl;
          // std::cout << vmin2 << " to " << vmax2 << std::endl;
          assert(hmin1 <= hmax1);
          assert(hmin2 <= hmax2);
          assert(vmin1 <= vmax1);
          assert(vmin2 <= vmax2);

          // std::cout << "h1: " << std::endl;
          // for (auto e : cr1.h_edges)
          //   std::cout << e.y << ", ";
          // std::cout << "\n";
          // std::cout << "h2: " << std::endl;
          // for (auto e : cr2.h_edges)
          //   std::cout << e.y << ", ";
          // std::cout << "\n";
          // std::cout << "v1: " << std::endl;
          // for (auto e : cr1.v_edges)
          //   std::cout << e.x << ", ";
          // std::cout << "\n";
          // std::cout << "h2: " << std::endl;
          // for (auto e : cr2.v_edges)
          //   std::cout << e.x << ", ";
          // std::cout << "\n";

          const auto h1 = std::lower_bound(
              cr1.h_edges.begin(), cr1.h_edges.end(), hmin2 - threshold,
              [](const h_edge& e, int v) { return e.y < v; });
          // std::cout << "find lower than" << hmin2 - threshold << " at "
          //           << std::distance(cr1.h_edges.begin(), h1) << std::endl;
          const auto h1m = std::upper_bound(
              cr1.h_edges.begin(), cr1.h_edges.end(), hmax2 + threshold,
              [](int v, const auto& e) { return v < e.y; });
          // std::cout << "find greater than" << hmax2 + threshold << " at "
          //           << std::distance(cr1.h_edges.begin(), h1m) << std::endl;
          const auto h2 = std::lower_bound(
              cr2.h_edges.begin(), cr2.h_edges.end(), hmin1 - threshold,
              [](const auto& e, int v) { return e.y < v; });
          const auto h2m = std::upper_bound(
              cr2.h_edges.begin(), cr2.h_edges.end(), hmax1 + threshold,
              [](int v, const auto& e) { return v < e.y; });
          const auto v1 = std::lower_bound(
              cr1.v_edges.begin(), cr1.v_edges.end(), vmin2 - threshold,
              [](const auto& e, int v) { return e.x < v; });
          const auto v1m = std::upper_bound(
              cr1.v_edges.begin(), cr1.v_edges.end(), vmax2 + threshold,
              [](int v, const auto& e) { return v < e.x; });
          const auto v2 = std::lower_bound(
              cr2.v_edges.begin(), cr2.v_edges.end(), vmin1 - threshold,
              [](const auto& e, int v) { return e.x < v; });
          const auto v2m = std::upper_bound(
              cr2.v_edges.begin(), cr2.v_edges.end(), vmax1 + threshold,
              [](int v, const auto& e) { return v < e.x; });
          assert(h1 != cr1.h_edges.end());
          assert(h2 != cr2.h_edges.end());
          assert(v1 != cr1.v_edges.end());
          assert(v2 != cr2.v_edges.end());
          // std::cout << "h1: from " << std::distance(cr1.h_edges.begin(), h1)
          //           << " to " << std::distance(cr1.h_edges.begin(), h1m)
          //           << ", totally " << cr1.h_edges.size() << std::endl;
          // std::cout << "h2: from " << std::distance(cr2.h_edges.begin(), h2)
          //           << " to " << std::distance(cr2.h_edges.begin(), h2m)
          //           << ", totally " << cr2.h_edges.size() << std::endl;
          // std::cout << "v1: from " << std::distance(cr1.v_edges.begin(), v1)
          //           << " to " << std::distance(cr1.v_edges.begin(), v1m)
          //           << ", totally " << cr1.v_edges.size() << std::endl;
          // std::cout << "v2: from " << std::distance(cr2.v_edges.begin(), v2)
          //           << " to " << std::distance(cr2.v_edges.begin(), v2m)
          //           << ", totally " << cr2.v_edges.size() << std::endl;
          run_space_check(&*h1, h1m - h1, &*h2, h2m - h2, &*v1, v1m - v1, &*v2,
                          v2m - v2, coord_bufferh1, coord_bufferh2,
                          coord_bufferv1, coord_bufferv2, check_resultsh,
                          check_resultsv, check_resultsh_host,
                          check_resultsv_host, threshold, graph_exec);
        }
        // int ovlp = 0;
        // for (auto& t : map) {
        //   if (t.second->interval.x_right >= e.interval.x_left and
        //       t.second->interval.x_left <= e.interval.x_right) {
        //     ++ovlp;
        //   }
        // }
        //   std::cout << i << ": [" << e.interval.x_left << ", " <<
        //   e.interval.x_right
        //             << "] overlaps with " << ovlp << ", now tree size is "
        //             << add - del << std::endl;
        // std::cout << e.intvl.mid() << std::endl;
        ++cnt;
        sum += add - del;
        total_ovlp += ovlp.size();
        // map.insert({e.interval.id, &e});
        tree.insert(e.intvl);
        ++add;
      } else {
        // map.erase(e.interval.id);
        tree.remove(e.intvl);
        ++del;
      }
    }
    t.pause();
  }
  std::cout << "total ovlp: " << total_ovlp << std::endl;
  std::cout << "Avg: " << double(sum) / cnt << std::endl;
}
}  // namespace odrc