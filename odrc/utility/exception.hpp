#pragma once

#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

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

// system errors

class error_category : public std::error_category {
 public:
  const char* name() const noexcept override { return "odrc"; }

  std::string message(int err_code) const override {
    switch (err_code) {
      default:
        return "unknown error";
    }
  }
};

class system_error : public std::system_error {
 public:
  system_error(std::error_code err_code) : std::system_error(err_code) {}

  system_error(std::error_code err_code, const std::string& what)
      : std::system_error(err_code, what) {}

  system_error(std::error_code err_code, const char* what)
      : std::system_error(err_code, what) {}

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  system_error(T ev, const std::error_category& err_category)
      : std::system_error(static_cast<int>(ev), err_category) {}

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  system_error(T                          ev,
               const std::error_category& err_category,
               const std::string&         what)
      : std::system_error(static_cast<int>(ev), err_category, what) {}

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  system_error(T ev, const std::error_category& err_category, const char* what)
      : std::system_error(static_cast<int>(ev), err_category, what) {}

  system_error(int err_code) : std::system_error(err_code, error_category()) {}

  system_error(int err_code, const std::string& what)
      : std::system_error(err_code, error_category(), what) {}

  system_error(int err_code, const char* what)
      : std::system_error(err_code, error_category(), what) {}
};

}  // namespace odrc
