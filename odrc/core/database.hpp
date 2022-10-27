#pragma once
#include <list>
#include <iostream>
#include <map>
#include <odrc/core/cell.hpp>
#include <odrc/utility/datetime.hpp>
#include <string>
#include <vector>

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
/*
struct horizontal_line {
  int         x_min;
  int         x_max;
  int         y;
  std::string name;
};
std::vector<std::string> inter_check( database *db) {
  std::list<horizontal_line> sorted;
  std::vector<std::string>   inter_cell;
  for (const auto& cel : db->cells) {
    auto i     = sorted.begin();
    bool y_min = true;
    bool y_max = true;
    while (y_max) {
      if (cel.mbr[2] >= i->y) {
        if (y_min) {
          sorted.insert(i,
              horizontal_line{cel.mbr[0], cel.mbr[1], cel.mbr[2], cel.name});
          y_min = false;
        }
        bool is_inter = !(cel.mbr[0] >= i->x_max || cel.mbr[1] <= i->x_min);
        if (is_inter) {
          inter_cell.emplace_back(std::string(cel.name) + "and" +
                             std::string(i->name) + "are overlapped.");
        }
      }
      if (cel.mbr[3] >= i->y) {
        sorted.insert(i,
            horizontal_line{cel.mbr[0], cel.mbr[1], cel.mbr[3], cel.name});
        y_max = false;
      }
      i++;
    }
  }
  return inter;
}*/
}  // namespace odrc::core