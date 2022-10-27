#include <odrc/utility/timer.hpp>

#include <doctest/doctest.h>

#include <chrono>
#include <fstream>
#include <thread>

void check_elapsed_time(const std::string& filename,
                        const int64_t      elapsed_time) {
  std::ifstream ifs(filename);
  if (!ifs) {
    throw std::runtime_error("Failed open file ");
  }

  std::string log_output;
  std::getline(ifs, log_output);

  CHECK(log_output.find(std::to_string(elapsed_time) + " ms") !=
        std::string::npos);
}

TEST_SUITE("[OpenDRC] odrc::timer tests") {
  TEST_CASE("test timer") {
    int64_t            sleep_time   = 2000;
    const std::string  log_filename = "log_timer_test.txt";
    odrc::util::logger logger(log_filename, odrc::util::log_level::trace,
                              false);
    {
      odrc::util::timer test_timer("test", &logger);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
    logger.flush();
    check_elapsed_time(log_filename, sleep_time);
    std::remove(log_filename.c_str());
  }
}
