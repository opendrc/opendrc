#include <odrc/algorithm/sequential_mode.hpp>

#include <cassert>
#include <iostream>

#include <odrc/core/cell.hpp>
#include <odrc/utility/logger.hpp>
#include <odrc/utility/timer.hpp>

namespace odrc {

void _check(odrc::core::database&         db,
            int                           layer,
            int                           i,
            int                           threshold,
            std::vector<core::violation>& vios) {
  check_distance(db.cells.at(i).upper_edges.at(layer),
                 db.cells.at(i).lower_edges.at(layer), threshold, vios);
  check_distance(db.cells.at(i).right_edges.at(layer),
                 db.cells.at(i).left_edges.at(layer), threshold, vios, false);
}

void width_check_seq(odrc::core::database&         db,
                     int                           layer,
                     int                           threshold,
                     std::vector<core::violation>& vios) {
  odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
  odrc::util::timer  width_check("width_check", logger);
  width_check.start();
  // get cell violations
  std::map<int, std::vector<core::violation>> intra_vios;
  for (auto i = 0UL; i < db.cells.size(); i++) {
    if (db.cells.at(i).is_touching(layer) and i != db.top_cell_id) {
      intra_vios.emplace(i, std::vector<core::violation>());
      _check(db, layer, i, threshold, intra_vios.at(i));
    }
  }
  width_check.pause();
  // convert cell violations to cell-ref violations
  get_ref_vios(db, intra_vios, vios);
}
}  // namespace odrc