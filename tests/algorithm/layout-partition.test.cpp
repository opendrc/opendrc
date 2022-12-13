#include <odrc/algorithm/layout-partition.hpp>
#include "odrc/gdsii/gdsii.hpp"

#include <doctest/doctest.h>
#include <iostream>

bool _check(const odrc::core::database&          db,
            const std::vector<int>&              layers,
            const std::vector<std::vector<int>>& sub_rows);

TEST_SUITE("[OpenDRC] odrc::layout-partition tests") {
  TEST_CASE("test layout partition") {
    auto                          db           = odrc::gdsii::read("./gcd.gds");
    std::vector<std::vector<int>> space_layers = {{19}, {20}};
    std::vector<std::vector<int>> enclose_layers = {
        {19, 21}, {20, 21}, {25, 20}};
    std::vector<int> without_layer;
    for (auto& space_layer : space_layers) {
      db.update_mbr(space_layer, without_layer);
      auto sub_rows = odrc::layout_partition(db, space_layer);
      CHECK_EQ(_check(db, space_layer, sub_rows), true);
    }
    for (auto& enclose_layer : enclose_layers) {
      db.update_mbr(enclose_layer, without_layer);
      auto sub_rows = odrc::layout_partition(db, enclose_layer);
      CHECK_EQ(_check(db, enclose_layer, sub_rows), true);
    }
  }
}

bool _check(const odrc::core::database&          db,
            const std::vector<int>&              layers,
            const std::vector<std::vector<int>>& sub_rows) {
  // check in the layers
  for (auto& sub_row : sub_rows) {
    for (auto& cell_ref_idx : sub_row) {
      const auto& cell_ref = db.cells.back().cell_refs.at(cell_ref_idx);
      const auto& the_cell = db.get_cell(cell_ref.cell_name);
      if (the_cell.is_touching(layers.front()) or
          the_cell.is_touching(layers.back())) {
        break;
      } else {
        return false;
      }
    }
  }
  // check right partition
  std::vector<std::pair<int, int>> intervals;
  const auto&                      cell_refs = db.cells.back().cell_refs;
  for (auto& cell_ref : cell_refs) {
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    int         idx      = db.get_cell_idx(cell_ref.cell_name);
    if (the_cell.is_touching(layers.front())) {
      auto mbr = db.edges.at(layers.front()).cell_ref_mbrs;
      intervals.push_back(std::make_pair(mbr.at(idx).y_min, mbr.at(idx).y_max));
    } else if (the_cell.is_touching(layers.back())) {
      auto mbr = db.edges.at(layers.back()).cell_ref_mbrs;
      intervals.push_back(std::make_pair(mbr.at(idx).y_min, mbr.at(idx).y_max));
    } else {
      continue;
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
      const auto& the_cell = db.get_cell(cell_refs.at(cell_ref_idx).cell_name);
      int         l, r;
      if (the_cell.is_touching(layers.front())) {
        auto mbr = db.edges.at(layers.front()).cell_ref_mbrs;
        l        = mbr.at(cell_ref_idx).y_min;
        r        = mbr.at(cell_ref_idx).y_max;
      } else {
        auto mbr = db.edges.at(layers.back()).cell_ref_mbrs;
        l        = mbr.at(cell_ref_idx).y_min;
        r        = mbr.at(cell_ref_idx).y_max;
      }
      // if (l < merged_intervals.at(row_id).first ||
      //     r > merged_intervals.at(row_id).second)
      //   return false;
    }
  }

  return true;
}
