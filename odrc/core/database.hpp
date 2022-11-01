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
    return cells.at(_name_to_idx.at(name));
  }
  const cell& get_cell(const std::string& name) const {
    return cells.at(_name_to_idx.at(name));
  }

  void update_map() {
    // update map if it's not up-to-date
    // do nothing if both containers have equal sizes
    for (auto i = _name_to_idx.size(); i < cells.size(); ++i) {
      _name_to_idx.emplace(cells.at(i).name, i);
    }
  }
  void update_depth_and_mbr() {
    for (auto& cell : cells) {
      if (cell.cell_refs.size() == 0) {
        cell.depth = 1;
      } else {
        int depth = 1;
        for (auto& cell_ref : cell.cell_refs) {
          auto& the_cell = get_cell(cell_ref.cell_name);
          depth = depth < the_cell.depth + 1 ? the_cell.depth + 1 : depth;
          cell_ref.mbr[0] = the_cell.mbr[0] + cell_ref.ref_point.x;
          cell_ref.mbr[1] = the_cell.mbr[1] + cell_ref.ref_point.x;
          cell_ref.mbr[2] = the_cell.mbr[2] + cell_ref.ref_point.y;
          cell_ref.mbr[3] = the_cell.mbr[3] + cell_ref.ref_point.y;
        }
        cell.depth = depth;
      }
    }
  }

  std::vector<cell> cells;

 private:
  std::unordered_map<std::string, int> _name_to_idx;
};

}  // namespace odrc::core