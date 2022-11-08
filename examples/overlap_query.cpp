#include <vector>

#include <odrc/core/interval_tree.hpp>
#include <odrc/gdsii/gdsii.hpp>
std::vector<std::pair<int, int>> overlap_query(odrc::core::database& db,
                                               const int&            layer) {
  std::vector<odrc::core::interval> sorted_edge;
  std::vector<std::pair<int, int>>  overlap_cells;

  // input data
  for (const auto& cell : db.cells) {
    if (cell.is_touching(layer)) {
      for (const auto& poly : cell.polygons) {
        if (layer == poly.layer) {
          sorted_edge.emplace_back(
              odrc::core::interval{poly.mbr[0], poly.mbr[1], poly.mbr[2],
                                   sorted_edge.size() / 2 + 1});
          sorted_edge.emplace_back(
              odrc::core::interval{poly.mbr[0], poly.mbr[1], poly.mbr[3],
                                   -sorted_edge.size() / 2 - 1});
        }
      }
      for (const auto& cell_ref : cell.cell_refs) {
        auto& the_cell = db.get_cell(cell_ref.cell_name);
        if (the_cell.is_touching(layer)) {
          int ref_x = cell_ref.ref_point.x;
          int ref_y = cell_ref.ref_point.y;
          for (const auto& poly : the_cell.polygons) {
            if (layer == poly.layer) {
              sorted_edge.emplace_back(odrc::core::interval{
                  poly.mbr[0] + ref_x, poly.mbr[1] + ref_x, poly.mbr[2] + ref_y,
                  sorted_edge.size() / 2 + 1});
              sorted_edge.emplace_back(odrc::core::interval{
                  poly.mbr[0] + ref_x, poly.mbr[1] + ref_x, poly.mbr[3] + ref_y,
                  -sorted_edge.size() / 2 - 1});
            }
          }
        }
      }
    }
  }
  std::sort(sorted_edge.begin(), sorted_edge.end(),
            [](const odrc::core::interval& a, const odrc::core::interval& b) {
              return a.y < b.y || (a.y == b.y && a.id > b.id);
            });
  odrc::core::interval_tree tree;
  for (int edge = 0; edge != sorted_edge.size(); edge++) {
    if (sorted_edge[edge].id > 0) {
      auto overlap_intervals =
          tree.get_intervals_overlapping_with(sorted_edge[edge]);
      overlap_cells.insert(overlap_cells.end(), overlap_intervals.begin(),
                           overlap_intervals.end());
      tree.add_interval(sorted_edge[edge]);
    } else {
      tree.delete_interval(sorted_edge[edge]);
    }
  }
  return overlap_cells;
}
int main(int argc, char* argv[]) {
  auto db = odrc::gdsii::read(argv[1]);
  db.update_depth_and_mbr();
  overlap_query(db, 2);
  return 0;
}
