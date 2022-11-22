#pragma once

#include <cuda_runtime.h>

namespace odrc::execution {
class sequenced_policy {};
class parallel_policy {};

class stream_executor {
 public:
  stream_executor() {}

 private:
  cudaStream_t _stream;
};

constexpr sequenced_policy seq;
constexpr parallel_policy  par;
}  // namespace odrc::execution