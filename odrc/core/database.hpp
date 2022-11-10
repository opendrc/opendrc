#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
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
    if (name == cells.back().name)
      return cells.back();
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
    // convert polygons to cells
    std::vector<cell> pcells;
    auto&             top_cell = cells.back();
    int               pi       = 0;
    for (auto& polygon : top_cell.polygons) {
      pcells.emplace_back();
      pcells.back().polygons.emplace_back(polygon);
      pcells.back().add_layer(polygon.layer);
      memcpy(pcells.back().mbr, polygon.mbr, sizeof(int) * 4);
      pcells.back().name = "polygon" + std::to_string(++pi);
      top_cell.cell_refs.emplace_back(pcells.back().name, coord{0, 0});
    }
    cells.insert(cells.end() - 1, pcells.begin(), pcells.end());
    for(int i = _name_to_idx.size()-1; i < cells.size(); ++i) {
      _name_to_idx[cells.at(i).name] = i;
    }

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
          if (the_cell.depth == 1) {  // lowest cells that only contains polys
            for (const auto& polygon : the_cell.polygons) {
              if (polygon.layer != 19)
                continue;
              const auto& points = polygon.points;
              for (auto i = 0; i < points.size() - 1; ++i) {
                if (points.at(i).x == points.at(i + 1).x) {  // v_edge
                  cell_ref.v_edges.emplace_back(
                      v_edge{points.at(i).x + cell_ref.ref_point.x,
                             points.at(i).y + cell_ref.ref_point.y,
                             points.at(i + 1).y + cell_ref.ref_point.y});
                } else {
                  cell_ref.h_edges.emplace_back(
                      h_edge{points.at(i).x + cell_ref.ref_point.x,
                             points.at(i + 1).x + cell_ref.ref_point.x,
                             points.at(i).y + cell_ref.ref_point.y});
                }
              }
              std::sort(
                  cell_ref.v_edges.begin(), cell_ref.v_edges.end(),
                  [](const auto& e1, const auto& e2) { return e1.x < e2.x; });
              std::sort(
                  cell_ref.h_edges.begin(), cell_ref.h_edges.end(),
                  [](const auto& e1, const auto& e2) { return e1.y < e2.y; });
            }
          }
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