#pragma once

#include <exception>
#include <stdexcept>
#include <string>

namespace odrc {

// logic errors

class not_implemented_error : public std::logic_error {
 public:
  not_implemented_error() : std::logic_error("Function not implemented.") {}
  not_implemented_error(const std::string& what_arg)
      : std::logic_error(what_arg) {}
  not_implemented_error(const char* what_arg) : std::logic_error(what_arg) {}
};

// runtime errors

class open_file_error : public std::runtime_error {
 public:
  open_file_error() : std::runtime_error("Open file error.") {}
  open_file_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
  open_file_error(const char* what_arg) : std::runtime_error(what_arg) {}
};

class invalid_file : public std::runtime_error {
 public:
  invalid_file() : std::runtime_error("Invalid file.") {}
  invalid_file(const std::string& what_arg) : std::runtime_error(what_arg) {}
  invalid_file(const char* what_arg) : std::runtime_error(what_arg) {}
};

}  // namespace odrc
