#include <odrc/algorithm/layout-partition.hpp>
#include "odrc/gdsii/gdsii.hpp"

#include <doctest/doctest.h>
#include <iostream>

bool _check(odrc::core::database&                db,
            const std::vector<int>&              layers,
            const std::vector<std::vector<int>>& sub_rows);

TEST_SUITE("[OpenDRC] odrc::layout-partition tests") {
  TEST_CASE("test layout partition") {
    auto                          db           = odrc::gdsii::read("./gcd.gds");
    std::vector<std::vector<int>> space_layers = {{19}, {20}};
    std::vector<std::vector<int>> enclose_layers = {
        {19, 21}, {20, 21}, {25, 20}};
    for (auto& space_layer : space_layers) {
      auto sub_rows = odrc::layout_partition(db, space_layer);
      CHECK_EQ(_check(db, space_layer, sub_rows), true);
    }
    for (auto& enclose_layer : enclose_layers) {
      auto sub_rows = odrc::layout_partition(db, enclose_layer);
      CHECK_EQ(_check(db, enclose_layer, sub_rows), true);
    }
  }
}

bool _check(odrc::core::database&                db,
            const std::vector<int>&              layers,
            const std::vector<std::vector<int>>& sub_rows) {
  // check in the layers
  for (auto& sub_row : sub_rows) {
    for (auto& cell_ref_idx : sub_row) {
      const auto& cell_ref = db.get_top_cell().cell_refs.at(cell_ref_idx);
      const auto& the_cell = db.get_cell(cell_ref.cell_name);
      if (not the_cell.is_touching(layers))
        return false;
    }
  }
  // check right partition
  std::vector<std::pair<int, int>> intervals;
  const auto&                      cell_refs = db.get_top_cell().cell_refs;
  for (auto& cell_ref : cell_refs) {
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (the_cell.is_touching(layers)) {
      auto mbr = cell_ref.cell_ref_mbr;
      intervals.emplace_back(mbr.y_min, mbr.y_max + 17);
    }
  }
  std::sort(intervals.begin(), intervals.end());

  std::vector<std::pair<int, int>> merged_intervals;
  int                              start = 0, end = 0;
  for (auto interval : intervals) {
    if (start == 0 && end == 0) {
      start = interval.first;
      end   = interval.second;
    } else if (interval.first > end) {
      merged_intervals.emplace_back(start, end);
      start = interval.first;
      end   = interval.second;
    } else
      end = std::max(end, interval.second);
  }
  merged_intervals.emplace_back(start, end);

  for (auto row_id = 0UL; row_id < sub_rows.size(); row_id++) {
    const auto& sub_row = sub_rows.at(row_id);
    for (auto cell_ref_idx : sub_row) {
      auto mbr = cell_refs.at(cell_ref_idx).cell_ref_mbr;
      int  l   = mbr.y_min;
      int  r   = mbr.y_max + 17;
      if (l < merged_intervals.at(row_id).first ||
          r > merged_intervals.at(row_id).second)
        return false;
    }
  }

  return true;
}
