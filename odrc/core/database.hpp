#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <odrc/core/cell.hpp>
#include <odrc/core/structs.hpp>
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
  const cell& get_cell(const std::string& name) const {
    return cells.at(_name_to_idx.at(name));
  }
  void erase() {
    cells.back().cell_refs.erase(
        cells.back().cell_refs.end() - cells.back().polygons.size(),
        cells.back().cell_refs.end());
    cells.erase(cells.end() - cells.back().polygons.size() - 1,
                cells.end() - 1);
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

  void update_mbr_and_edge(std::vector<int> layers) {
    // convert polygons to cells
    std::vector<cell> pcells;
    int               pi       = 0;
    auto&             top_cell = cells.back();
    pcells.reserve(top_cell.polygons.size());
    for (auto& polygon : top_cell.polygons) {
      pcells.emplace_back();
      pcells.back().polygons.emplace_back(polygon);
      pcells.back().add_layer(polygon.layer);
      pcells.back().name = "polygon" + std::to_string(++pi);
      top_cell.cell_refs.emplace_back(pcells.back().name, coord{0, 0});
    }
    cells.insert(cells.end() - 1, std::move(pcells.begin()),
                 std::move(pcells.end()));
    update_map();
    for (const auto& layer : layers) {
      cell_edges.emplace(layer, edges());
      auto& _edge    = cell_edges.at(layer);
      auto& cell_ref = cells.back().cell_refs;
      for (auto i = 0UL; i < cell_ref.size(); i++) {
        const auto& mbr      = get_cell(cell_ref.at(i).cell_name).mbr;
        const auto& the_cell = get_cell(cell_ref.at(i).cell_name);
        auto&       ref_mbr  = cell_ref.at(i).cell_ref_mbr;
        ref_mbr.x_min        = mbr.x_min + cell_ref.at(i).ref_point.x;
        ref_mbr.x_max        = mbr.x_max + cell_ref.at(i).ref_point.x;
        ref_mbr.y_min        = mbr.y_min + cell_ref.at(i).ref_point.y;
        ref_mbr.y_max        = mbr.y_max + cell_ref.at(i).ref_point.y;
        for (const auto& polygon : the_cell.polygons) {
          const auto& points = polygon.points;
          _edge.h_edges.emplace_back();
          _edge.v_edges.emplace_back();
          for (auto j = 0UL; j < points.size() - 1; ++j) {
            if (points.at(j).x == points.at(j + 1).x) {  // v_edge
              _edge.v_edges.back().emplace_back(
                  v_edge{points.at(j).x + cell_ref.at(i).ref_point.x,
                         points.at(j).y + cell_ref.at(i).ref_point.y,
                         points.at(j + 1).y + cell_ref.at(i).ref_point.y});

            } else {
              _edge.h_edges.back().emplace_back(
                  h_edge{points.at(j).x + cell_ref.at(i).ref_point.x,
                         points.at(j + 1).x + cell_ref.at(i).ref_point.x,
                         points.at(j).y + cell_ref.at(i).ref_point.y});
            }
          }
        }
        std::sort(_edge.h_edges.back().begin(), _edge.h_edges.back().end(),
                  [](const auto& e1, const auto& e2) { return e1.y < e2.y; });
        std::sort(_edge.v_edges.back().begin(), _edge.v_edges.back().end(),
                  [](const auto& e1, const auto& e2) { return e1.x < e2.x; });
      }
    }
  }

  std::vector<cell>    cells;
  std::map<int, edges> cell_edges;

 private:
  std::unordered_map<std::string, int> _name_to_idx;
};

}  // namespace odrc::core