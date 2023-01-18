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
  int                  version     = -1;
  int                  top_cell_id = -1;
  std::string          name;
  odrc::util::datetime mtime;
  odrc::util::datetime atime;
  double               dbu_in_user_unit;
  double               dbu_in_meter;

  // layout
  cell& create_cell() { return cells.emplace_back(); }

  int get_cell_idx(const std::string& name) const {
    return _name_to_idx.at(name);
  }

  cell& get_cell(const std::string& name) {
    return cells.at(_name_to_idx.at(name));
  }

  void update_top_cell_id() {
    int size = 0;
    for (auto i = 0UL; i < cells.size(); i++) {
      if (cells.at(i).cell_refs.size() > size) {
        size        = cells.at(i).cell_refs.size();
        top_cell_id = i;
      }
    }
  }

  cell& get_top_cell() {
    if (top_cell_id == -1) {
      update_top_cell_id();
    }
    return cells.at(top_cell_id);
  }

  void update_map() {
    // update map if it's not up-to-date
    // do nothing if both containers have equal sizes
    for (auto i = _name_to_idx.size(); i < cells.size(); ++i) {
      _name_to_idx.emplace(cells.at(i).name, i);
    }
  }

  void convert_polygon_to_cell() {
    const auto num_top_cell_polygons = get_top_cell().polygons.size();
    // convert polygons to cells
    for (int i = 0; i < num_top_cell_polygons; i++) {
      auto&       the_new_cell = create_cell();
      const auto& the_polygon  = get_top_cell().polygons.at(i);
      the_new_cell.name        = "polygon" + std::to_string(i);
      the_new_cell.polygons.emplace_back(the_polygon);
      the_new_cell.update_mbr(the_polygon.mbr);
      the_new_cell.add_layer(the_polygon.layer);
      auto& the_new_cell_ref = get_top_cell().cell_refs.emplace_back(
          "polygon" + std::to_string(i), coord{0, 0});
      the_new_cell_ref.update_mbr(the_polygon.mbr);
    }
    update_map();
  }

  template <typename C>
  void create_edges(C& c, cell& the_cell) {
    for (int layer = 0; layer < 64; layer++) {
      if (the_cell.is_touching(layer)) {
        c.left_edges.emplace(layer, std::vector<orthogonal_edge>());
        c.right_edges.emplace(layer, std::vector<orthogonal_edge>());
        c.upper_edges.emplace(layer, std::vector<orthogonal_edge>());
        c.lower_edges.emplace(layer, std::vector<orthogonal_edge>());
      }
    }
  }

  template <typename C>
  void add_edges(C&           c,
                 int          layer,
                 const coord& point,
                 const coord& next_point,
                 const coord& ref_point) {
    auto x1 = point.x + ref_point.x;
    auto y1 = point.y + ref_point.y;
    auto x2 = next_point.x + ref_point.x;
    auto y2 = next_point.y + ref_point.y;
    if (x1 == x2) {
      if (y1 > y2) {
        c.right_edges[layer].emplace_back(orthogonal_edge{y2, y1, x1});
      } else {
        c.left_edges[layer].emplace_back(orthogonal_edge{y1, y2, x1});
      }
    } else {
      if (x1 < x2) {
        c.upper_edges[layer].emplace_back(orthogonal_edge{x1, x2, y1});
      } else {
        c.lower_edges[layer].emplace_back(orthogonal_edge{x2, x1, y1});
      }
    }
  }

  void update_edges() {
    for (auto i = 0UL; i < cells.size(); i++) {
      if (i == top_cell_id)
        continue;
      create_edges(cells.at(i), cells.at(i));
      for (const auto& polygon : cells.at(i).polygons) {
        const auto& points = polygon.points;
        for (auto j = 0UL; j < points.size() - 1; ++j) {
          add_edges(cells.at(i), polygon.layer, points.at(j), points.at(j + 1),
                    coord{0, 0});
        }
      }
    }
    for (auto& cell_ref : get_top_cell().cell_refs) {
      auto& cell = get_cell(cell_ref.cell_name);
      create_edges(cell_ref, cell);
      for (const auto& polygon : cell.polygons) {
        const auto& points = polygon.points;
        for (auto j = 0UL; j < points.size() - 1; ++j) {
          add_edges(cell_ref, polygon.layer, points.at(j), points.at(j + 1),
                    cell_ref.ref_point);
        }
      }
    }
  }
  std::vector<cell> cells;

 private:
  std::unordered_map<std::string, int> _name_to_idx;
};
}  // namespace odrc::core