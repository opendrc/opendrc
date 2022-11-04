#include <vector>

#include <odrc/core/interval_tree.hpp>
#include <odrc/gdsii/gdsii.hpp>

namespace examples {

void get_polygon_edge(const odrc::core::cell*            cells,
                      std::vector<odrc::core::interval>* sorted_edge,
                      int                                layer,
                      int                                ref_x,
                      int                                ref_y,
                      int*                               id) {
  for (const auto& poly : cells->polygons) {
    if (poly.layer == layer) {
      sorted_edge->emplace_back(odrc::core::interval{
          poly.mbr[0] + ref_x, poly.mbr[1] + ref_x, poly.mbr[2] + ref_y, *id});
      sorted_edge->emplace_back(odrc::core::interval{
          poly.mbr[0] + ref_x, poly.mbr[1] + ref_x, poly.mbr[3] + ref_y, -*id});
      id++;
    }
  }
}

void get_edge(odrc::core::database&              db,
              std::vector<odrc::core::interval>* sorted_edge,
              const odrc::core::cell*            cells,
              int                                layer,
              int*                               id) {
  get_polygon_edge(cells, sorted_edge, layer, 0, 0, id);
  if (cells->depth > 1) {
    for (const auto& cell_ref : cells->cell_refs) {
      auto the_cell = db.get_cell(cell_ref.cell_name);
      if (the_cell.layers == layer) {
        int ref_x = cell_ref.ref_point.x;
        int ref_y = cell_ref.ref_point.y;
        get_polygon_edge(&the_cell, sorted_edge, layer, ref_x, ref_y, id);
      }
      get_edge(db, sorted_edge, &the_cell, layer, id);
    }
  }
}

std::vector<std::pair<int, int>> overlap_query(odrc::core::database& db,
                                               const int&            layer) {
  std::vector<odrc::core::interval> sorted_edge;
  std::vector<std::pair<int, int>>  overlap_cells;
  int                               id = 1;

  // input data
  for (const auto& cells : db.cells) {
    get_edge(db, &sorted_edge, &cells, layer, &id);
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
        tree.get_overlapping_intervals(&sorted_edge[edge], 0, &overlap_cells);
        tree.add_interval(&sorted_edge[edge], 0);
      }
    } else {
      tree.delete_interval(&sorted_edge[edge], 0);
    }
  }
  return overlap_cells;
}

}  // namespace examples

int main(int argc, char* argv[]) {
  auto db = odrc::gdsii::read(argv[1]);
  db.update_depth_and_mbr();
  examples::overlap_query(db, 2);
}
