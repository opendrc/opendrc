#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <odrc/core/cell.hpp>
#include <odrc/utility/datetime.hpp>

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
  cell& get_cell(const std::string& name) {
    // update map if it's not up-to-date
    // do nothing if both containers have equal sizes
    for (auto i = _name_to_idx.size(); i < cells.size(); ++i) {
      _name_to_idx.emplace(cells.at(i).name, i);
    }
    return cells.at(_name_to_idx.at(name));
  }

  std::vector<cell> cells;

 private:
  std::unordered_map<std::string, int> _name_to_idx;
};

}  // namespace odrc::core