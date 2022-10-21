#pragma once

#include <exception>
#include <stdexcept>

namespace odrc {

class not_implemented_error : public std::logic_error {
 public:
  not_implemented_error(const char* what_arg = "Function not implemented.")
      : std::logic_error(what_arg) {}
};

class invalid_file : public std::runtime_error {
 public:
  invalid_file(const char* what_arg = "Invalid file.")
      : std::runtime_error(what_arg) {}
};

}  // namespace odrc
