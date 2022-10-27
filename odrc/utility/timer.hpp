#pragma once

#include <odrc/utility/logger.hpp>

#include <chrono>
#include <iostream>
#include <string>

namespace odrc::util {

class timer {
 public:
  timer(std::string tag, logger* time_logger) {
    _tag              = tag;
    _logger           = time_logger;
    _begin_time_point = std::chrono::high_resolution_clock::now();
  }

  ~timer() {
    const auto _end_time_point = std::chrono::high_resolution_clock::now();
    int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             _end_time_point - _begin_time_point)
                             .count();

    _logger->info("Timer", "[{}] {} ms", _tag, elapsed_ms);
  }

 private:
  std::string                                                 _tag;
  logger*                                                     _logger;
  std::chrono::time_point<std::chrono::high_resolution_clock> _begin_time_point;
};

}  // namespace odrc::util
