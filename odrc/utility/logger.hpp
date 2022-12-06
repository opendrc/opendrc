#pragma once

#include <map>
#include <memory>
#include <utility>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace odrc::util {
enum class log_level { trace, debug, info, warn, error, critical, off };

class logger {
 public:
  logger(const std::string& log_filename,
         const log_level&   log_level,
         bool               output_to_console = true) {
    _sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        log_filename.c_str()));
    if (output_to_console)
      _sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

    _logger = std::make_shared<spdlog::logger>("OpenDRC", _sinks.begin(),
                                               _sinks.end());

    _logger->set_pattern("[%D %H:%M:%S] [%n] [%L] [thread %t] %v");
    _logger->set_level(_log_level_map.at(log_level));
  }

  ~logger() {}

  void set_level(const log_level& log_level) {
    _logger->set_level(_log_level_map.at(log_level));
  }

  template <typename... Args>
  inline void trace(const std::string& source,
                    const std::string& message,
                    const Args&... args) {
    _logger->trace("[" + source + "] " + message, args...);
  }

  template <typename... Args>
  inline void debug(const std::string& source,
                    const std::string& message,
                    const Args&... args) {
    _logger->debug("[" + source + "] " + message, args...);
  }

  template <typename... Args>
  inline void info(const std::string& source,
                   const std::string& message,
                   const Args&... args) {
    _logger->info("[" + source + "] " + message, args...);
  }

  template <typename... Args>
  inline void warn(const std::string& source,
                   const std::string& message,
                   const Args&... args) {
    _logger->warn("[" + source + "] " + message, args...);
  }

  template <typename... Args>
  inline void error(const std::string& source,
                    const std::string& message,
                    const Args&... args) {
    _logger->error("[" + source + "] " + message, args...);
  }

  template <typename... Args>
  void critical(const std::string& source,
                const std::string& message,
                const Args&... args) {
    _logger->critical("[" + source + "] " + message, args...);
  }

  void flush() { _logger->flush(); }

 private:
  std::shared_ptr<spdlog::logger> _logger;
  std::vector<spdlog::sink_ptr>   _sinks;

  const std::map<log_level, spdlog::level::level_enum> _log_level_map{
      {log_level::trace, spdlog::level::level_enum::trace},
      {log_level::debug, spdlog::level::level_enum::debug},
      {log_level::info, spdlog::level::level_enum::info},
      {log_level::warn, spdlog::level::level_enum::warn},
      {log_level::error, spdlog::level::level_enum::err},
      {log_level::critical, spdlog::level::level_enum::critical},
      {log_level::off, spdlog::level::level_enum::off},
  };
};
}  // namespace odrc::util
