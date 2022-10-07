#include <odrc/infrastructure/logger/logger.hpp>

#include <doctest/doctest.h>

#include <fstream>
#include <vector>

std::size_t count_lines(const std::string& filename) {
  std::ifstream ifs(filename);
  if (!ifs) {
    throw std::runtime_error("Failed open file ");
  }

  std::string line;
  size_t      counter = 0;
  while (std::getline(ifs, line))
    counter++;
  return counter;
}

void require_message_count(const std::string& filename,
                           const std::size_t  messages) {
  if (strlen(spdlog::details::os::default_eol) == 0) {
    REQUIRE(count_lines(filename) == 1);
  } else {
    REQUIRE(count_lines(filename) == messages);
  }
}

TEST_SUITE("[OpenDRC] odrc::logger tests") {
  const std::string  log_filename = "log_test.txt";
  odrc::core::logger logger(log_filename, odrc::core::log_level::TRACE);
  TEST_CASE("test info level") {
    logger.trace("doctest", "Test message level {}", "TRACE");
    logger.debug("doctest", "Test message level {}", "DEBUG");
    logger.info("doctest", "Test message level {}", "INFO");
    logger.warn("doctest", "Test message level {}", "WARN");
    logger.error("doctest", "Test message level {}", "ERROR");
    logger.critical("doctest", "Test message level {}", "CRITICAL");
    logger.flush();
    require_message_count(log_filename, 6);
  }
}