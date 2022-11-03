#include <odrc/core/interval_tree.cpp>
#include <odrc/core/interval_tree.hpp>

#include <vector>
namespace examples {
std::vector<std::pair<int, int>> overlap_query(odrc::core::database& db,
                                               const int&            layer) {
  std::vector<odrc::core::interval> sorted_edge;
  std::vector<std::pair<int, int>>  overlap_cells;
  int                               id = 1;
  // input data
  for (const auto& cell : db.cells) {
    for (const auto& poly : cell.polygons) {
      if (poly.layer == layer) {
        sorted_edge.emplace_back(
            odrc::core::interval{poly.mbr[0], poly.mbr[1], poly.mbr[2], id});
        sorted_edge.emplace_back(
            odrc::core::interval{poly.mbr[0], poly.mbr[1], poly.mbr[3], -id});
        id++;
      }
    }
    for (const auto& cell_ref : cell.cell_refs) {
      auto cell_refed = db.get_cell(cell_ref.cell_name);
      if (cell_refed.layers == layer) {
        sorted_edge.emplace_back(
            odrc::core::interval{cell_refed.mbr[0] + cell_ref.ref_point.x,
                                 cell_refed.mbr[1] + cell_ref.ref_point.x,
                                 cell_refed.mbr[2] + cell_ref.ref_point.y, id});
        sorted_edge.emplace_back(odrc::core::interval{
            cell_refed.mbr[0] + cell_ref.ref_point.x,
            cell_refed.mbr[1] + cell_ref.ref_point.x,
            cell_refed.mbr[3] + cell_ref.ref_point.y, -id});
        id++;
      }
    }
  }
  std::sort(sorted_edge.begin(), sorted_edge.end(),
            [](const odrc::core::interval& a, const odrc::core::interval& b) {
              return a.y < b.y;
            });

  odrc::core::interval_tree tree;
  for (int edge = 0; edge != sorted_edge.size(); edge++) {
    if (sorted_edge[edge].id > 0) {
      if (edge == 0) {
        tree.add_node(&sorted_edge[edge]);
      } else {
        auto overlap_cell = tree.overlap_interval_query(&sorted_edge[edge], 0);
        overlap_cells.insert(overlap_cells.end(), overlap_cell.begin(),
                             overlap_cell.end());
        tree.add_interval(&sorted_edge[edge], 0);
      }
    } else {
      tree.delete_interval(&sorted_edge[edge], 0);
    }
  }
  return overlap_cells;
}
}  // namespace examples