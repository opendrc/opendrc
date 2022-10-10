#pragma once

#include <exception>

namespace odrc {

enum class error {
  no_error              = 0,
  not_implemented_error = 10,
  invalid_file_error    = 20
};

class exception : public std::exception {
 public:
  exception(error err) throw() { _error = err; }

  virtual const char* what() const throw() { return "OpenDRC Exception"; }

  error error_code() const throw() { return _error; }

 private:
  error _error;
};

class not_implemented_exception : public exception {
 public:
  not_implemented_exception(const char* message = nullptr) throw()
      : exception(error::not_implemented_error) {
    _message = message;
  }

  const char* what() const throw() {
    return (_message == nullptr ? "NOT_IMPLEMENTED_ERROR" : _message);
  }

 private:
  const char* _message;
};

class invalid_file_exception : public exception {
 public:
  invalid_file_exception(const char* message = nullptr) throw()
      : exception(error::invalid_file_error) {
    _message = message;
  }

  const char* what() const throw() {
    return (_message == nullptr ? "INVALID_FILE_ERROR" : _message);
  }

 private:
  const char* _message;
};

}  // namespace odrc
