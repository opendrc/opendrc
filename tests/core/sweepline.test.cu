#include <odrc/core/sweepline.hpp>

#include <vector>

#include <cuda_runtime.h>
#include <doctest/doctest.h>

#include <odrc/core/execution.hpp>

TEST_SUITE("[OpenDRC] odrc::core parallel sweepline tests") {
  TEST_CASE("test naive parallel prefix sum") {
    std::vector<int> events{0, 1, 2, 3};
    std::vector<int> result(4, -1);
    std::vector<int> golden{0, 1, 3, 6};
    int*             events_dev;
    int*             result_dev;
    cudaMalloc((void**)&events_dev, sizeof(int) * 4);
    cudaMalloc((void**)&result_dev, sizeof(int) * 4);
    cudaMemcpy(events_dev, events.data(), sizeof(int) * 4,
               cudaMemcpyHostToDevice);
    odrc::core::sweepline(odrc::execution::par, events_dev, events_dev + 4, result_dev,
                          [] __device__(int* l, int* r) -> int {
                            int sum = 0;
                            for (int* p = l; p != r; ++p)
                              sum += *p;
                            return sum;
                          });
    cudaMemcpy(result.data(), result_dev, sizeof(int) * 4,
               cudaMemcpyDeviceToHost);
    CHECK_EQ(result, golden);
  }
}