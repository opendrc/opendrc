#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <odrc/core/cell.hpp>
#include <odrc/core/common_structs.hpp>
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

  void update_map() {
    // update map if it's not up-to-date
    // do nothing if both containers have equal sizes
    for (auto i = _name_to_idx.size(); i < cells.size(); ++i) {
      _name_to_idx.emplace(cells.at(i).name, i);
    }
  }
  void erase_mbr() {
    cells.back().cell_refs.erase(
        cells.back().cell_refs.end() - cells.back().polygons.size(),
        cells.back().cell_refs.end());
    cells.erase(cells.end() - cells.back().polygons.size() - 1,
                cells.end() - 1);
    for (auto& cell : cells) {
      cell.depth = -1;
    }
  };
  void update_mbr(std::vector<int> layers, std::vector<int> without_layer) {
    // convert polygons to cells
    std::vector<cell> pcells;
    int               pi = 0;
    for (auto& polygon : cells.back().polygons) {
      pcells.emplace_back();
      pcells.back().polygons.emplace_back(polygon);
      pcells.back().add_layer(polygon.layer);
      memcpy(pcells.back().mbr1, polygon.mbr1, sizeof(int) * 4);
      pcells.back().name = "polygon" + std::to_string(++pi);
      cells.back().cell_refs.emplace_back(pcells.back().name, coord{0, 0});
    }
    cells.insert(cells.end() - 1, pcells.begin(), pcells.end());
    for (auto i = _name_to_idx.size() - 1; i < cells.size(); ++i) {
      _name_to_idx[cells.at(i).name] = i;
    }
    for (const auto& layer : layers) {
      edges.emplace(layer, edge());
      auto& _edge = edges.at(layer);
      for (const auto& cell : cells) {
        _edge.mbrs.emplace_back();
        for (const auto& polygon : cell.polygons) {
          _edge.mbrs.back().x_min =
              std::min(_edge.mbrs.back().x_min, polygon.mbr1[0]);
          _edge.mbrs.back().x_max =
              std::max(_edge.mbrs.back().x_min, polygon.mbr1[1]);
          _edge.mbrs.back().y_min =
              std::min(_edge.mbrs.back().x_min, polygon.mbr1[2]);
          _edge.mbrs.back().y_max =
              std::max(_edge.mbrs.back().x_min, polygon.mbr1[3]);
        }
      }
      const auto& cell_ref = cells.back().cell_refs;
      for (int i = 0; i < cell_ref.size(); i++) {
        int idx = get_cell_idx(cell_ref.at(i).cell_name);
        _edge.cell_ref_mbrs.emplace_back(mbr{
            _edge.mbrs.at(idx).x_min + cell_ref.at(i).ref_point.x,
            _edge.mbrs.at(idx).x_max + cell_ref.at(i).ref_point.x,
            _edge.mbrs.at(idx).y_min + cell_ref.at(i).ref_point.y,
            _edge.mbrs.at(idx).y_max + cell_ref.at(i).ref_point.y,
        });
        for (const auto& polygon : cells.at(idx).polygons) {
          const auto& points = polygon.points;
          _edge.h_edges.emplace_back();
          _edge.v_edges.emplace_back();
          for (auto i = 0; i < points.size() - 1; ++i) {
            if (points.at(i).x == points.at(i + 1).x) {  // v_edge
              _edge.v_edges.back().emplace_back(
                  v_edge{points.at(i).x + cell_ref.at(idx).ref_point.x,
                         points.at(i).y + cell_ref.at(idx).ref_point.y,
                         points.at(i + 1).y + cell_ref.at(idx).ref_point.y});

            } else {
              _edge.h_edges.back().emplace_back(
                  h_edge{points.at(i).x + cell_ref.at(idx).ref_point.x,
                         points.at(i + 1).x + cell_ref.at(idx).ref_point.x,
                         points.at(i).y + cell_ref.at(idx).ref_point.y});
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

  std::vector<cell>   cells;
  std::map<int, edge> edges;

 private:
  std::unordered_map<std::string, int> _name_to_idx;
};

}  // namespace odrc::core