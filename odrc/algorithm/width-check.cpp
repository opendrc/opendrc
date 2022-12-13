#include <odrc/algorithm/width-check.hpp>

#include <cassert>
#include <iostream>

#include <odrc/core/common_structs.hpp>

namespace odrc {
using odrc::core::polygon;

void _check_polygon(const polygon&             poly,
                    int                        threshold,
                    std::vector<check_result>& vios) {
  int num = poly.points.size() - 1;
  // Loop through all pairs of edges
  for (int i = 0; i < num; ++i) {
    int e11x = poly.points.at(i).x;
    int e11y = poly.points.at(i).y;
    int e12x = poly.points.at(i + 1).x;
    int e12y = poly.points.at(i + 1).y;
    for (int j = i + 2; j < num; ++j) {
      // Check if the two edges are parallel
      int e21x = poly.points.at(j).x;
      int e21y = poly.points.at(j).y;
      int e22x = poly.points.at(j + 1).x;
      int e22y = poly.points.at(j + 1).y;
      // Check the distance between the two edges
      int dist = 0;
      if (e11x == e12x) {
        dist = abs(e11x - e21x);
      } else {
        dist = abs(e11y - e21y);
      }
      if (dist < threshold) {
        // Check if the edges overlap
        bool overlap = false;
        if (e11x == e12x) {
          overlap = e11x < e21x ? (e22y < e21y and e12y > e22y and e21y > e11y)
                                : (e21y < e22y and e12y < e22y and e21y < e11y);
        } else {
          overlap = e11y < e21y ? (e22x < e21x and e12x > e22x and e21x > e11x)
                                : (e21x < e22x and e12x < e22x and e21x < e11x);
        }
        if (overlap) {
          vios.emplace_back(check_result{e11x, e11y, e12x, e12y, e21x, e21y,
                                         e22x, e22y, true});
        }
      }
    }
  }
}

void width_check_seq(const odrc::core::database& db,
                     int                         layer,
                     int                         threshold,
                     std::vector<check_result>&  vios) {
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