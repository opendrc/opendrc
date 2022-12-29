#include <odrc/algorithm/sequential_mode.hpp>

#include <cassert>
#include <iostream>
#include "odrc/core/cell.hpp"

namespace odrc {
using odrc::core::polygon;

void _check_polygon(const polygon& poly,
                    int            threshold,
                    rule_type      ruletype,
                    int            ref_x,
                    int            ref_y,
                    std::string name,
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
      bool is_vlt = is_violation(edge1, edge2, threshold, ruletype);
      if (is_h_edge and is_vlt) {
        vios.emplace_back(violation_information{
            core::edge{start_point1 + ref_x, distance1 + ref_y,
                       end_point1 + ref_x, distance1 + ref_y},
            core::edge{start_point2 + ref_x, distance2 + ref_y,
                       end_point2 + ref_x, distance2 + ref_y}});
        std::cout <<name<<std::endl;
        //   std::cout << "layer" << poly.layer << std::endl;
        //   std::cout << "mbr: " << poly.mbr.x_min << " " << poly.mbr.x_max <<
        //   " "
        //             << poly.mbr.y_min << " " << poly.mbr.y_max << std::endl;
        //   for (auto i = 0; i <poly.points.size() ; ++i) {
        //     std::cout<<"x: "<< poly.points[i].x<<" y:"<<
        //     poly.points[i].y<<std::endl;
        //  }
      } else if ((!is_h_edge) and is_vlt) {
        vios.emplace_back(violation_information{
            core::edge{distance1 + ref_x, start_point1 + ref_y,
                       distance1 + ref_x, end_point1 + ref_y},
            core::edge{distance2 + ref_x, start_point2 + ref_y,
                       distance2 + ref_x, end_point2 + ref_y}});
        std::cout <<name<<std::endl;
      }
    }
  }
}

void width_check_seq(const odrc::core::database&         db,
                     int                                 layer,
                     int                                 threshold,
                     rule_type                           ruletype,
                     std::vector<violation_information>& vios) {
  // result memoization
  std::unordered_map<std::string, int> checked_results;

  static int checked_poly = 0;
  static int saved_poly   = 0;
  static int rotated      = 0;
  static int magnified    = 0;
  static int reflected    = 0;

  for (const auto& cell_ref : db.cells.back().cell_refs) {
    int   local_poly = 0;
    auto& cell       = db.cells.at(db.get_cell_idx(cell_ref.cell_name));
    for (const auto& polygon : cell.polygons) {
      if (polygon.layer != layer) {
        continue;
      }
      ++local_poly;
      _check_polygon(polygon, threshold, ruletype, cell_ref.ref_point.x,
                     cell_ref.ref_point.y,cell_ref.cell_name, vios);
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