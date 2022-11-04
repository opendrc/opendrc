#include <cassert>
#include <odrc/algorithm/width-check.hpp>

#include <iostream>

namespace odrc {

using odrc::core::polygon;

void _check_polygon(const polygon& poly, int threshold) {
  for (int i = 0; i < poly.points.size() - 1; ++i) {
    for (int j = i + 2; j < poly.points.size() - 1; ++j) {
      int  e11x         = poly.points.at(i).x;
      int  e11y         = poly.points.at(i).y;
      int  e12x         = poly.points.at(i + 1).x;
      int  e12y         = poly.points.at(i + 1).y;
      int  e21x         = poly.points.at(j).x;
      int  e21y         = poly.points.at(j).y;
      int  e22x         = poly.points.at(j + 1).x;
      int  e22y         = poly.points.at(j + 1).y;
      bool is_violation = false;
      // width check
      if (e11x == e12x) {  // vertical
        if (e11x < e21x) {
          // e12 e21
          // e11 e22
          bool is_inside_to_inside   = e11y < e12y and e21y > e22y;
          bool is_too_close          = e21x - e11x < threshold;
          bool is_projection_overlap = e11y < e21y and e22y < e12y;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        } else {
          // e22 e11
          // e21 e12
          bool is_inside_to_inside   = e21y < e22y and e11y > e21y;
          bool is_too_close          = e11x - e21x < threshold;
          bool is_projection_overlap = e21y < e11y and e12y < e22y;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        }
      } else {  // horizontal
        if (e11y < e22y) {
          // e21 e22
          // e12 e11
          bool is_inside_to_inside   = e11x > e12x and e21x < e22x;
          bool is_too_close          = e21y - e11y < threshold;
          bool is_projection_overlap = e21x < e11x and e12x < e22x;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        } else {
          // e11 e12
          // e22 e21
          bool is_inside_to_inside   = e21x > e22x and e11x < e12x;
          bool is_too_close          = e11y - e21y < threshold;
          bool is_projection_overlap = e11x < e21x and e22x < e12x;
          is_violation =
              is_inside_to_inside and is_too_close and is_projection_overlap;
        }
      }
    }
  }
}

void width_check_cpu(const odrc::core::database& db, int layer, int threshold) {
  // result memoization
  std::unordered_map<std::string, int> checked_results;

  static int checked_poly = 0;
  static int saved_poly   = 0;
  static int rotated      = 0;
  static int magnified    = 0;
  static int reflected    = 0;

  for (const auto& cell : db.cells) {
    if (not cell.is_touching(layer)) {
      continue;
    }
    int local_poly = 0;
    for (const auto& polygon : cell.polygons) {
      if (polygon.layer != layer) {
        continue;
      }
      ++local_poly;
      _check_polygon(polygon, threshold);
    }
    for (const auto& cell_ref : cell.cell_refs) {
      // should have been checked
      if (!db.get_cell(cell_ref.cell_name).is_touching(layer)) {
        continue;
      }
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
  std::cout << "checked: " << checked_poly << ", saved: " << saved_poly
            << "(mag: " << magnified << ", ref: " << reflected
            << ", rot: " << rotated << ")\n";
}
}  // namespace odrc