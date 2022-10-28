#include <odrc/utility/timer.hpp>

#include <doctest/doctest.h>

#include <chrono>
#include <fstream>
#include <thread>

void check_elapsed_time(const std::string& filename,
                        const int          sleep_count,
                        const int          reset_count,
                        const int64_t      sleep_time) {
  std::ifstream ifs(filename);
  if (!ifs) {
    throw std::runtime_error("Failed open file ");
  }

  std::string log_output;
  for (int i = 0; i < reset_count; ++i) {
    std::getline(ifs, log_output);
    CHECK(log_output.find(std::to_string(sleep_time) + " ms") !=
          std::string::npos);
  }

  std::getline(ifs, log_output);
  const int64_t expect_time = (sleep_count - reset_count) * sleep_time;
  CHECK(log_output.find(std::to_string(expect_time) + " ms") !=
        std::string::npos);
}

TEST_SUITE("[OpenDRC] odrc::timer tests") {
  TEST_CASE("test timer") {
    int64_t            sleep_time   = 2000;
    int                sleep_count  = 10;
    int                reset_count  = 5;
    const std::string  log_filename = "log_timer_test.txt";
    odrc::util::logger logger(log_filename, odrc::util::log_level::trace,
                              false);
    {
      odrc::util::timer test_timer("test", &logger);
      for (int i = 0; i < sleep_count; ++i) {
        test_timer.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        test_timer.pause();
        if (i < reset_count)
          test_timer.reset("reset " + std::to_string(i));
      }
    }
    logger.flush();
    check_elapsed_time(log_filename, sleep_count, reset_count, sleep_time);
    std::remove(log_filename.c_str());
  }
}
