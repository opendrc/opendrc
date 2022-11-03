#pragma once

#include <odrc/utility/logger.hpp>

#include <chrono>
#include <iostream>
#include <string>

namespace odrc::util {

class timer {
 public:
  timer(std::string tag, logger& time_logger) {
    _tag    = tag;
    _logger = &time_logger;
  }

  ~timer() { _logger->info("Timer", "[{}] {} ms", _tag, elapsed_ms); }

  void start() {
    _begin_time_point = std::chrono::high_resolution_clock::now();
  }

  void pause() {
    const auto _end_time_point = std::chrono::high_resolution_clock::now();
    elapsed_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
                      _end_time_point - _begin_time_point)
                      .count();
  }

  void reset(std::string message = "") {
    _logger->info("Timer", "[{}] {} {} ms", _tag, message, elapsed_ms);
    elapsed_ms = 0;
  }

  int64_t get_elapsed() { return elapsed_ms; }

 private:
  std::string                                                 _tag;
  logger*                                                     _logger;
  int64_t                                                     elapsed_ms = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> _begin_time_point;
};

}  // namespace odrc::util
