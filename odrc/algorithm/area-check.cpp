#include <cassert>
#include <odrc/algorithm/area-check.hpp>

#include <iostream>

namespace odrc {

using odrc::core::polygon;

void _check_polygon_area(const polygon& poly, int threshold) {
  int         area = 0;
  const auto& points =
      poly.points;  // TODO: I need to ensure the sequence of points is correct
  for (auto i = 0UL; i < points.size(); ++i) {
    auto j = (i + 1) % points.size();
    area += points.at(i).x * points.at(j).y - points.at(j).x * points.at(i).y;
  }
  area = abs(area / 2);
  if (area > threshold) {
    for (auto& p : points) {
      printf("(%d,%d),", p.x, p.y);
    }
    printf("area: %d %d\n ", area, area < threshold);
  }
}

void area_check_cpu(const odrc::core::database& db, int layer, int threshold) {
  // result memorization
  std::unordered_map<std::string, int> checked_results;

  static int checked_poly = 0;
  static int saved_poly   = 0;

  for (const auto& cell : db.cells) {
    if (not cell.is_touching(layer))
      continue;

    int local_poly = 0;
    for (const auto& polygon : cell.polygons) {
      if (polygon.layer != layer) {
        continue;
      }
      ++local_poly;
      _check_polygon_area(polygon, threshold);
    }

    for (const auto& cell_ref : cell.cell_refs) {
      if (!db.get_cell(cell_ref.cell_name).is_touching(layer)) {
        continue;
      }
      auto cell_checked = checked_results.find(cell_ref.cell_name);
      assert(cell_checked != checked_results.end());
      if (cell_checked != checked_results.end()) {
        saved_poly += cell_checked->second;
      }
    }

    checked_results.emplace(cell.name, local_poly);
    checked_poly += local_poly;
  }
}

}  // namespace odrc