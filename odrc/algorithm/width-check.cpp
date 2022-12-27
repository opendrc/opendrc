#include <odrc/algorithm/sequential_mode.hpp>

#include <cassert>
#include <iostream>
#include "odrc/core/cell.hpp"

namespace odrc {
using odrc::core::polygon;

void _check_polygon(const polygon&                      poly,
                    int                                 threshold,
                    std::vector<violation_information>& vios) {
  int         num   = poly.points.size() - 1;
  const auto& point = poly.points;
  // Loop through all pairs of edges
  for (auto i = 0; i < num; ++i) {
    bool is_h_edge    = point.at(i).x == point.at(i + 1).x;
    int  distance1    = is_h_edge ? point.at(i).x : point.at(i).y;
    int  start_point1 = is_h_edge ? point.at(i).y : point.at(i).x;
    int  end_point1   = is_h_edge ? point.at(i + 1).y : point.at(i + 1).x;
    for (int j = i + 2; j < num; ++j) {
      // Check if the two edges are parallel
      int distance2    = is_h_edge ? point.at(j).x : point.at(j).y;
      int start_point2 = is_h_edge ? point.at(j).y : point.at(j).x;
      int end_point2   = is_h_edge ? point.at(j + 1).y : point.at(j + 1).x;
      std::tuple<int, int, int> edge1{start_point1, end_point1, distance1};
      std::tuple<int, int, int> edge2{start_point2, end_point2, distance2};
      bool is_violation =is_spacing_violation(edge1, edge2, threshold);
      if (is_h_edge and is_violation) {
        vios.emplace_back(violation_information{
            core::edge{start_point1, distance1, end_point1, distance1},
            core::edge{start_point2, distance2, end_point2, distance1}});
      } else if ((!is_h_edge) and is_violation) {
        vios.emplace_back(violation_information{
            core::edge{distance1, start_point1, distance1, end_point1},
            core::edge{distance2, start_point2, distance2, end_point2}});
      }
    }
  }
}

void width_check_seq(const odrc::core::database&         db,
                     int                                 layer,
                     int                                 threshold,
                     std::vector<violation_information>& vios) {
  // result memoization
  std::unordered_map<std::string, int> checked_results;

  static int checked_poly = 0;
  static int saved_poly   = 0;
  static int rotated      = 0;
  static int magnified    = 0;
  static int reflected    = 0;

  for (const auto& cell : db.cells) {
    int local_poly = 0;
    for (const auto& polygon : cell.polygons) {
      if (polygon.layer != layer) {
        continue;
      }
      ++local_poly;
      _check_polygon(polygon, threshold, vios);
    }
    for (const auto& cell_ref : cell.cell_refs) {
      // should have been checked
      auto cell_checked = checked_results.find(cell_ref.cell_name);
      assert(cell_checked != checked_results.end());
      if (cell_checked != checked_results.end() and
          !cell_ref.trans.is_magnified and !cell_ref.trans.is_reflected and
          !cell_ref.trans.is_rotated) {
        saved_poly += cell_checked->second;
      } else {
        if (cell_ref.trans.is_magnified) {
          magnified += cell_checked->second;
        }
        if (cell_ref.trans.is_reflected) {
          reflected += cell_checked->second;
        }
        if (cell_ref.trans.is_rotated) {
          rotated += cell_checked->second;
        }
      }
    }
    checked_results.emplace(cell.name, local_poly);
    checked_poly += local_poly;
  }
}
}  // namespace odrc