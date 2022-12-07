#include <odrc/utility/logger.hpp>

#include <doctest/doctest.h>

#include <cstdio>
#include <fstream>

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

void check_message_count(const std::string& filename,
                         const std::size_t  messages) {
  if (strlen(spdlog::details::os::default_eol) == 0) {
    CHECK(count_lines(filename) == 1);
  } else {
    CHECK(count_lines(filename) == messages);
  }
}

TEST_SUITE("[OpenDRC] odrc::logger tests") {
  const std::string  log_filename = "log_test.txt";
  odrc::util::logger logger(log_filename, odrc::util::log_level::trace, false);
  TEST_CASE("test info level") {
    logger.trace("doctest", "Test message level {}", "TRACE");
    logger.debug("doctest", "Test message level {}", "DEBUG");
    logger.info("doctest", "Test message level {}", "INFO");
    logger.warn("doctest", "Test message level {}", "WARN");
    logger.error("doctest", "Test message level {}", "ERROR");
    logger.critical("doctest", "Test message level {}", "CRITICAL");
    logger.flush();
    check_message_count(log_filename, 6);
    std::remove(log_filename.c_str());
  }
}