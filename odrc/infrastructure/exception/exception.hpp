#pragma once

#include <exception>

namespace odrc::utility {

typedef enum odrc_error {
  ERROR_NONE            = 0,
  ERROR_NOT_IMPLEMENTED = 10,
  ERROR_INVALID_FILE    = 20
} odrc_error;

class odrc_exception : public std::exception {
 public:
  odrc_exception(odrc_error err) throw() { _error = err; }

  virtual const char* what() const throw() { return "OpenDRC Exception"; }

  odrc_error error_code() const throw() { return _error; }

 private:
  odrc_error _error;
};

class not_implemented_exception : public odrc_exception {
 public:
  not_implemented_exception(const char* message = 0) throw()
      : odrc_exception(ERROR_NOT_IMPLEMENTED) {
    _message = message;
  }

  const char* what() const throw() {
    return (_message == 0 ? "ERROR_NOT_IMPLEMENTED" : _message);
  }

 private:
  const char* _message;
};

class invaild_file_exception : public odrc_exception {
 public:
  invaild_file_exception(const char* message = 0) throw()
      : odrc_exception(ERROR_INVALID_FILE) {
    _message = message;
  }

  const char* what() const throw() {
    return (_message == 0 ? "ERROR_INVALID_FILE" : _message);
  }

 private:
  const char* _message;
};
}  // namespace odrc::utility