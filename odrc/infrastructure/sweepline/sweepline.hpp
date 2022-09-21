#pragma once

#include <stdexcept>
#include <type_traits>

#include <odrc/infrastructure/execution.hpp>

namespace odrc::core {

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
    throw std::logic_error("Parallel sweepline is not implemented.");
  }
}

}  // namespace odrc::core