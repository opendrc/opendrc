#include <odrc/infrastructure/sweepline/sweepline.hpp>

#include <vector>

#include <doctest/doctest.h>

#include <odrc/infrastructure/execution.hpp>

TEST_SUITE("[OpenDRC] odrc::sweepline tests") {
  TEST_CASE("test trivial sequenced copy") {
    std::vector<int> events{0, 1, 2, 3};
    std::vector<int> result;
    odrc::core::sweepline(odrc::execution::seq, events.begin(), events.end(),
                          &result, [](int i, auto v) { v->emplace_back(i); });
    CHECK_EQ(events, result);
  }
  TEST_CASE("test unimplmented parallel sweepline") {
    CHECK_THROWS_AS(odrc::core::sweepline(odrc::execution::par, 0, 0,
                                          static_cast<void*>(nullptr), 0),
                    std::logic_error);
  }
}