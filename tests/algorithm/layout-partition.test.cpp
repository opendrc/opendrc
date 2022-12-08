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
      db.update_depth_and_mbr(space_layer.front(), without_layer);
      auto sub_rows = odrc::layout_partition(db, space_layer);
      CHECK_EQ(_check(db, space_layer, sub_rows), true);
    }
    for (auto& enclose_layer : enclose_layers) {
      db.update_depth_and_mbr(enclose_layer, without_layer);
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

      bool is_touching = false;
      for (auto& layer : layers) {
        if (the_cell.is_touching(layer)) {
          is_touching = true;
          break;
        }
      }
      if (not is_touching) {
        return false;
      }
    }
  }
  // check right partition
  std::vector<std::pair<int, int>> intervals;
  const auto&                      cell_refs = db.cells.back().cell_refs;
  for (auto& cell_ref : cell_refs) {
    const auto& the_cell    = db.get_cell(cell_ref.cell_name);
    bool        is_touching = false;
    for (auto& layer : layers) {
      if (the_cell.is_touching(layer)) {
        is_touching = true;
        break;
      }
    }
    if (not is_touching)
      continue;
    if (the_cell.is_touching(layers.front())) {
      intervals.push_back(std::make_pair(cell_ref.mbr1[2], cell_ref.mbr1[3]));
    } else {
      intervals.push_back(std::make_pair(cell_ref.mbr2[2], cell_ref.mbr2[3]));
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
        l = cell_refs.at(cell_ref_idx).mbr1[2];
        r = cell_refs.at(cell_ref_idx).mbr1[3];
      } else {
        l = cell_refs.at(cell_ref_idx).mbr2[2];
        r = cell_refs.at(cell_ref_idx).mbr2[3];
      }
      if (l < merged_intervals.at(row_id).first ||
          r > merged_intervals.at(row_id).second)
        return false;
    }
  }

  return true;
}
