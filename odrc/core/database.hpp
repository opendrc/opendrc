#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <odrc/core/cell.hpp>
#include <odrc/utility/datetime.hpp>
namespace odrc {
struct check_result {
  int  e11x;
  int  e11y;
  int  e12x;
  int  e12y;
  int  e21x;
  int  e21y;
  int  e22x;
  int  e22y;
  bool is_violation = false;
};
enum class rule_type {
  area,
  enclosure,
  extension,
  geometry,
  length,
  lup,
  overlap,
  recommend,       // may be unused
  spacing_both,    // default,both horizontal and vertical edges
  spacing_h_edge,  // only horizontal edges
  spacing_v_edge,  // only vertical edges
  spacing_corner,  // corner-corner
  spacing_center,  // center-center
  spacing_tip,     // tip-tip
  spacing_lup,     // latch-upobject
  sram,
  width,
  aux_not_bend,
  aux_is_rectilinear,
  aux_ensure
};
}  // namespace odrc

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
  void update_depth_and_mbr(int layer, std::vector<int> without_layer) {
    // convert polygons to cells
    std::vector<cell> pcells;
    auto&             top_cell = cells.back();
    int               pi       = 0;
    for (auto& polygon : top_cell.polygons) {
      pcells.emplace_back();
      pcells.back().polygons.emplace_back(polygon);
      pcells.back().add_layer(polygon.layer);
      memcpy(pcells.back().mbr1, polygon.mbr1, sizeof(int) * 4);
      pcells.back().name = "polygon" + std::to_string(++pi);
      top_cell.cell_refs.emplace_back(pcells.back().name, coord{0, 0});
    }
    cells.insert(cells.end() - 1, pcells.begin(), pcells.end());
    for (long unsigned int i = _name_to_idx.size() - 1; i < cells.size(); ++i) {
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
          if (the_cell.depth == 1) {  // lowest cells that only contains polys
            for (const auto& polygon : the_cell.polygons) {
              if (polygon.layer != layer)
                continue;
              if (std::find(without_layer.begin(), without_layer.end(),
                            polygon.layer) != without_layer.end())
                continue;
              cell_ref.mbr1[0]   = std::min(polygon.mbr1[0], cell_ref.mbr1[0]);
              cell_ref.mbr1[1]   = std::max(polygon.mbr1[1], cell_ref.mbr1[1]);
              cell_ref.mbr1[2]   = std::min(polygon.mbr1[2], cell_ref.mbr1[2]);
              cell_ref.mbr1[3]   = std::max(polygon.mbr1[3], cell_ref.mbr1[3]);
              const auto& points = polygon.points;
              for (long unsigned int i = 0; i < points.size() - 1; ++i) {
                if (points.at(i).x == points.at(i + 1).x) {  // v_edge
                  cell_ref.v_edges1.emplace_back(
                      v_edge{points.at(i).x + cell_ref.ref_point.x,
                             points.at(i).y + cell_ref.ref_point.y,
                             points.at(i + 1).y + cell_ref.ref_point.y});
                } else {
                  cell_ref.h_edges1.emplace_back(
                      h_edge{points.at(i).x + cell_ref.ref_point.x,
                             points.at(i + 1).x + cell_ref.ref_point.x,
                             points.at(i).y + cell_ref.ref_point.y});
                }
              }
            }
          }
          std::sort(cell_ref.v_edges1.begin(), cell_ref.v_edges1.end(),
                    [](const auto& e1, const auto& e2) { return e1.x < e2.x; });
          std::sort(cell_ref.h_edges1.begin(), cell_ref.h_edges1.end(),
                    [](const auto& e1, const auto& e2) { return e1.y < e2.y; });
          cell_ref.mbr1[0] += cell_ref.ref_point.x;
          cell_ref.mbr1[1] += cell_ref.ref_point.x + 17;
          cell_ref.mbr1[2] += cell_ref.ref_point.y;
          cell_ref.mbr1[3] += cell_ref.ref_point.y + 17;
        }
        cell.depth = depth;
      }
    }
  }

  void update_depth_and_mbr(std::vector<int> layers,
                            std::vector<int> without_layer) {
    // convert polygons to cells
    std::vector<cell> pcells;
    auto&             top_cell = cells.back();
    int               pi       = 0;
    for (auto& polygon : top_cell.polygons) {
      pcells.emplace_back();
      pcells.back().polygons.emplace_back(polygon);
      pcells.back().add_layer(polygon.layer);
      memcpy(pcells.back().mbr1, polygon.mbr1, sizeof(int) * 4);
      pcells.back().name = "polygon" + std::to_string(++pi);
      top_cell.cell_refs.emplace_back(pcells.back().name, coord{0, 0});
    }
    cells.insert(cells.end() - 1, pcells.begin(), pcells.end());
    for (int i = _name_to_idx.size() - 1; i < cells.size(); ++i) {
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
          if (the_cell.depth == 1) {  // lowest cells that only contains polys
            for (const auto& polygon : the_cell.polygons) {
              if (polygon.layer != layers.front() and
                  polygon.layer != layers.back())
                continue;
              if (std::find(without_layer.begin(), without_layer.end(),
                            polygon.layer) != without_layer.end())
                continue;
              if (polygon.layer == layers.front()) {
                cell_ref.mbr1[0] = std::min(polygon.mbr1[0], cell_ref.mbr1[0]);
                cell_ref.mbr1[1] = std::max(polygon.mbr1[1], cell_ref.mbr1[1]);
                cell_ref.mbr1[2] = std::min(polygon.mbr1[2], cell_ref.mbr1[2]);
                cell_ref.mbr1[3] = std::max(polygon.mbr1[3], cell_ref.mbr1[3]);
              } else {
                cell_ref.mbr2[0] = std::min(polygon.mbr1[0], cell_ref.mbr2[0]);
                cell_ref.mbr2[1] = std::max(polygon.mbr1[1], cell_ref.mbr2[1]);
                cell_ref.mbr2[2] = std::min(polygon.mbr1[2], cell_ref.mbr2[2]);
                cell_ref.mbr2[3] = std::max(polygon.mbr1[3], cell_ref.mbr2[3]);
              }
              const auto&          points = polygon.points;
              std::vector<v_edge>& vedges = polygon.layer == layers.front()
                                                ? cell_ref.v_edges1
                                                : cell_ref.v_edges2;
              std::vector<h_edge>& hedges = polygon.layer == layers.front()
                                                ? cell_ref.h_edges1
                                                : cell_ref.h_edges2;
              for (auto i = 0; i < points.size() - 1; ++i) {
                if (points.at(i).x == points.at(i + 1).x) {  // v_edge
                  vedges.emplace_back(
                      v_edge{points.at(i).x + cell_ref.ref_point.x,
                             points.at(i).y + cell_ref.ref_point.y,
                             points.at(i + 1).y + cell_ref.ref_point.y});
                } else {
                  hedges.emplace_back(
                      h_edge{points.at(i).x + cell_ref.ref_point.x,
                             points.at(i + 1).x + cell_ref.ref_point.x,
                             points.at(i).y + cell_ref.ref_point.y});
                }
              }
            }
            std::sort(
                cell_ref.v_edges1.begin(), cell_ref.v_edges1.end(),
                [](const auto& e1, const auto& e2) { return e1.x < e2.x; });
            std::sort(
                cell_ref.v_edges2.begin(), cell_ref.v_edges2.end(),
                [](const auto& e1, const auto& e2) { return e1.x < e2.x; });

            std::sort(
                cell_ref.h_edges1.begin(), cell_ref.h_edges1.end(),
                [](const auto& e1, const auto& e2) { return e1.y < e2.y; });
            std::sort(
                cell_ref.h_edges2.begin(), cell_ref.h_edges2.end(),
                [](const auto& e1, const auto& e2) { return e1.y < e2.y; });
          }

          cell_ref.mbr1[0] += cell_ref.ref_point.x;
          cell_ref.mbr1[1] += cell_ref.ref_point.x + 17;
          cell_ref.mbr1[2] += cell_ref.ref_point.y;
          cell_ref.mbr1[3] += cell_ref.ref_point.y + 17;
          cell_ref.mbr2[0] += cell_ref.ref_point.x;
          cell_ref.mbr2[1] += cell_ref.ref_point.x + 17;
          cell_ref.mbr2[2] += cell_ref.ref_point.y;
          cell_ref.mbr2[3] += cell_ref.ref_point.y + 17;
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