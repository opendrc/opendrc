#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
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
  int   get_cell_idx(const std::string& name) const {
    return _name_to_idx.at(name);
  }
  cell& get_cell(const std::string& name) {
    return cells.at(_name_to_idx.at(name));
  }
  cell& get_top_cell() {
    unsigned long idx=0;
    int size=0;
    for (auto i = 0UL; i < cells.size(); i++) {
      if(cells.at(i).cell_refs.size()>size)
      {
        size=cells.at(i).cell_refs.size();
        idx=i;
      }
    }
    return cells.at(idx);
  };
  int get_top_cell_idx(){
    unsigned long idx=0;
    int size=0;
    for (auto i = 0UL; i < cells.size(); i++) {
      if(cells.at(i).cell_refs.size()>size)
      {
        size=cells.at(i).cell_refs.size();
        idx=i;
      }
    }
    return idx;
  };
  void update_map() {
    // update map if it's not up-to-date
    // do nothing if both containers have equal sizes
    if (_name_to_idx.size() == 0) {
      for (auto i = 0UL; i < cells.size(); ++i) {
        _name_to_idx.emplace(cells.at(i).name, i);
      }
    } else {
      for (auto i = _name_to_idx.size() - 1; i < cells.size(); ++i) {
        _name_to_idx[cells.at(i).name] = i;
      }
    }
  }

  void convert_polygon_to_cell() {
    // convert polygons to cells
    cells.insert(cells.end() - 1, cell());
    auto _cell  = cells.end() - 2;
    _cell->name = "polygon";
    for (auto& polygon : get_top_cell().polygons) {
      _cell->polygons.emplace_back(polygon);
      _cell->add_layer(polygon.layer);
    }
    get_top_cell().cell_refs.emplace_back("polygon", coord{0, 0});
    update_map();
  }

  void update_edges() {
    for (auto num = 0UL; num < cells.size() ; num++) {
      if(num==get_top_cell_idx())
        continue;
      for (int layer = 0; layer < 64; layer++) {
        if (cells.at(num).is_touching(layer)) {
          cells.at(num).left_edges.emplace(layer,
                                           std::vector<orthogonal_edge>());
          cells.at(num).right_edges.emplace(layer,
                                            std::vector<orthogonal_edge>());
          cells.at(num).upper_edges.emplace(layer,
                                            std::vector<orthogonal_edge>());
          cells.at(num).lower_edges.emplace(layer,
                                            std::vector<orthogonal_edge>());
        }
      }
      for (const auto& polygon : cells.at(num).polygons) {
        const auto& points = polygon.points;
        for (auto j = 0UL; j < points.size() - 1; ++j) {
          if (points.at(j).x == points.at(j).x and
              points.at(j).y > points.at(j + 1).y) {
            cells.at(num).right_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j+1).y, points.at(j ).y,
                                points.at(j).x});
          } else if (points.at(j).x == points.at(j).x and
                     points.at(j).y < points.at(j + 1).y) {
            cells.at(num).left_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j).y, points.at(j+1 ).y,
                                points.at(j).x});
          } else if (points.at(j).y == points.at(j).y and
                     points.at(j).x < points.at(j + 1).x) {
            cells.at(num).upper_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j).x, points.at(j+1 ).x,
                                points.at(j).y});
          } else if (points.at(j).y == points.at(j).y and
                     points.at(j).x > points.at(j + 1).x) {
            cells.at(num).lower_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j+1).x, points.at(j ).x,
                                points.at(j).y});
          }
        }
      }
    }
    for (auto& cell_ref : get_top_cell().cell_refs) {
      const auto& cell = get_cell(cell_ref.cell_name);
      for (int layer = 0; layer < 64; layer++) {
        if (cell.is_touching(layer)) {
          cell_ref.left_edges.emplace(layer, std::vector<orthogonal_edge>());
          cell_ref.right_edges.emplace(layer, std::vector<orthogonal_edge>());
          cell_ref.upper_edges.emplace(layer, std::vector<orthogonal_edge>());
          cell_ref.lower_edges.emplace(layer, std::vector<orthogonal_edge>());
        }
      }
      for (const auto& polygon : cell.polygons) {
        const auto& points = polygon.points;
        for (auto j = 0UL; j < points.size() - 1; ++j) {
          if (points.at(j).x == points.at(j).x and
              points.at(j).y > points.at(j + 1).y) {
            cell_ref.right_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j+1).y + cell_ref.ref_point.y,
                                points.at(j ).y + cell_ref.ref_point.y,
                                points.at(j).x + cell_ref.ref_point.x});
          } else if (points.at(j).x == points.at(j).x and
                     points.at(j).y < points.at(j + 1).y) {
            cell_ref.left_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j).y + cell_ref.ref_point.y,
                                points.at(j+1 ).y + cell_ref.ref_point.y,
                                points.at(j).x + cell_ref.ref_point.x});
          } else if (points.at(j).y == points.at(j).y and
                     points.at(j).x < points.at(j + 1).x) {
            cell_ref.upper_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j).x + cell_ref.ref_point.x,
                                points.at(j +1).x + cell_ref.ref_point.x,
                                points.at(j).y + cell_ref.ref_point.y});
          } else if (points.at(j).y == points.at(j).y and
                     points.at(j).x > points.at(j + 1).x) {
            cell_ref.lower_edges[polygon.layer].emplace_back(
                orthogonal_edge{points.at(j+1).x + cell_ref.ref_point.x,
                                points.at(j ).x + cell_ref.ref_point.x,
                                points.at(j).y + cell_ref.ref_point.y});
          }
        }
      }
    }
  }
  std::vector<cell> cells;

 private:
  std::unordered_map<std::string, int> _name_to_idx;
};
}  // namespace odrc::core