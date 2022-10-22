#pragma once

#include <string>
#include <vector>

#include <odrc/core/cell.hpp>

namespace odrc::core {

class database {
 public:
  // meta info
  int                  version = -1;
  std::string          name;
  odrc::util::datetime mtime;
  odrc::util::datetime atime;
  double               dbu_in_user_unit;
  double               dbu_in_meter;

  // layout
  cell& create_cell() { return cells.emplace_back(); }

  std::vector<cell> cells;
};
}  // namespace odrc::core