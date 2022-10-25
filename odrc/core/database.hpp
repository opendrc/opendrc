#pragma once

<<<<<<< HEAD
#include <algorithm>
#include <iostream>
#include <map>
#include <odrc/core/cell.hpp>
#include <odrc/utility/datetime.hpp>
#include <string>
#include <vector>

=======
#include <string>
#include <vector>

#include <odrc/core/cell.hpp>
#include <odrc/utility/datetime.hpp>

>>>>>>> 39871cac67c41f4a4d8467f7735ff4cdefdb6cc0
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

<<<<<<< HEAD
  std::map<std::string, int64_t>          cell_layer;
  std::map<std::string, std::vector<int>> min_bound_rect;
=======
>>>>>>> 39871cac67c41f4a4d8467f7735ff4cdefdb6cc0
  // layout
  cell& create_cell() { return cells.emplace_back(); }

  std::vector<cell> cells;
<<<<<<< HEAD

  void min_bound();
  void layer_sort();
=======
>>>>>>> 39871cac67c41f4a4d8467f7735ff4cdefdb6cc0
};
}  // namespace odrc::core