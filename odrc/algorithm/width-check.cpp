#include <odrc/algorithm/sequential_mode.hpp>

#include <cassert>
#include <iostream>
#include "odrc/core/cell.hpp"

namespace odrc {

void _check_cell(odrc::core::database&               db,
                 int                                 layer,
                 int                                 num,
                 int                                 threshold,
                 std::vector<violation_information>& vios) {
  check_distance(db.cells.at(num).upper_edges.at(layer),
                 db.cells.at(num).lower_edges.at(layer), threshold, vios);
  check_distance(db.cells.at(num).right_edges.at(layer),
                 db.cells.at(num).left_edges.at(layer), threshold, vios);
}

void width_check_seq(odrc::core::database&               db,
                     int                                 layer,
                     int                                 threshold,
                     std::vector<violation_information>& vios) {
  // get cell violations
  std::map<int, std::vector<violation_information>> intra_vios;
  for (auto num = 0UL; num < db.cells.size(); num++) {
    if (db.cells.at(num).is_touching(layer) and num != db.get_top_cell_idx()) {
      intra_vios.emplace(num, std::vector<violation_information>());
      _check_cell(db, layer, num, threshold, intra_vios.at(num));
    }
  }
  //convert cell violations to cell-ref violations
  get_ref_vios(db, intra_vios, vios);
}
}  // namespace odrc