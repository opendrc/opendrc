#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <cassert>
#include <stdexcept>
#include <type_traits>

#include <odrc/infrastructure/execution.hpp>

namespace odrc::core {

#ifdef __CUDACC__
template <typename EventPointIt, typename Status, typename Fold>
__global__ void _build_prefix(EventPointIt first,
                              EventPointIt last,
                              Status*      status,
                              Fold         fold) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (first + i < last) {
    status[i] = fold(first, first + i + 1);
  }
}

template <typename EventPointIt, typename Status, typename Fold>
void _sweepline_par(EventPointIt first,
                    EventPointIt last,
                    Status*      status,
                    Fold         fold) {
  int n = std::distance(first, last);
  _build_prefix<<<(n + 127) / 128, 128>>>(first, last, status, fold);
  cudaDeviceSynchronize();
  cudaError_t cuda_status = cudaGetLastError();
  assert(cuda_status == cudaSuccess);
}
#endif

template <typename EventPointIt, typename Status, typename Op>
void _sweepline(EventPointIt first, EventPointIt last, Status* status, Op op) {
  for (auto it = first; it != last; ++it) {
    op(*it, status);
  }
}

template <typename ExecutionPolicy,
          typename EventPointIt,
          typename Status,
          typename Op>
void sweepline(ExecutionPolicy&& policy,
               EventPointIt      first,
               EventPointIt      last,
               Status*           status,
               Op                op) {
  if constexpr (std::is_same_v<
                    std::remove_cv_t<std::remove_reference_t<decltype(policy)>>,
                    odrc::execution::sequenced_policy>) {
    _sweepline(first, last, status, op);
  } else {
#ifdef __CUDACC__
    _sweepline_par(first, last, status, op);
#else
    throw std::logic_error("Parallel sweepline is not implemented.");
#endif
  }
}

}  // namespace odrc::core