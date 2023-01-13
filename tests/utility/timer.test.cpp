#include <odrc/utility/timer.hpp>

#include <doctest/doctest.h>

#include <chrono>
#include <fstream>
#include <thread>

void check_elapsed_time(std::vector<int64_t>& elapsed_times,
                        const int             sleep_count,
                        const int             reset_count,
                        const int64_t         sleep_time) {
  bool in_range;
  for (int i = 0; i < reset_count; ++i) {
    in_range =
        elapsed_times[i] >= sleep_time && elapsed_times[i] <= sleep_time * 1.1;
    CHECK(in_range);
  }

  const int64_t expect_time = (sleep_count - reset_count) * sleep_time;

  in_range = elapsed_times.back() >= expect_time &&
             elapsed_times.back() <= expect_time * 1.1;
  CHECK(in_range);
}

void check_log_file(const std::string&          filename,
                    const int&                  reset_count,
                    const std::vector<int64_t>& elapsed_times) {
  std::ifstream ifs(filename);
  if (!ifs) {
    throw std::runtime_error("Failed open file ");
  }

  std::string log_output;
  for (auto i = 0UL; i < reset_count; ++i) {
    std::getline(ifs, log_output);
    CHECK(log_output.find(std::to_string(elapsed_times[i]) + " ms") !=
          std::string::npos);
  }

  std::getline(ifs, log_output);
  CHECK(log_output.find(std::to_string(elapsed_times.back()) + " ms") !=
        std::string::npos);
}

TEST_SUITE("[OpenDRC] odrc::timer tests") {
  TEST_CASE("test timer") {
    int64_t              sleep_time  = 100;
    int                  sleep_count = 5;
    int                  reset_count = 2;
    std::vector<int64_t> elapsed_times;
    const std::string    log_filename = "log_timer_test.txt";
    odrc::util::logger   logger(log_filename, odrc::util::log_level::trace,
                              false);
    {
      odrc::util::timer test_timer("test", logger);
      for (int i = 0; i < sleep_count; ++i) {
        test_timer.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        test_timer.pause();

        if (i < reset_count) {
          elapsed_times.push_back(test_timer.get_elapsed());
          test_timer.reset("reset " + std::to_string(i));
        }
      }
      elapsed_times.push_back(test_timer.get_elapsed());
    }
    logger.flush();

    check_elapsed_time(elapsed_times, sleep_count, reset_count, sleep_time);
    check_log_file(log_filename, reset_count, elapsed_times);
    std::remove(log_filename.c_str());
  }
}
