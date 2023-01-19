#pragma once

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <vector>

#include <odrc/core/database.hpp>

// give sub-rows, to indicate the cell_id and the corresponding cell_x,y

// layout_partition() declaration: partitions the whole cells in a specific
// layer into a set of sub-rows(sub-regions).
namespace odrc {
inline std::vector<std::vector<int>> layout_partition(odrc::core::database& db,
                                                      std::vector<int> layers,
                                                      int threshold = 18) {
  using interval_t      = std::pair<int, int>;
  const auto& cell_refs = db.get_top_cell().cell_refs;

  // get unique cell y-axis coordinates and store intervals need to merge
  std::unordered_set<int> coordinates;
  std::vector<interval_t> intervals;
  std::vector<int>        cell_ref_ids;

  for (auto id = 0UL; id < cell_refs.size(); ++id) {
    const auto& cell_ref = cell_refs[id];
    const auto& the_cell = db.get_cell(cell_ref.cell_name);

    if (the_cell.is_touching(layers)) {
      cell_ref_ids.emplace_back(id);
      auto& mbr = cell_ref.cell_ref_mbr;
      coordinates.insert(mbr.y_min);
      coordinates.insert(mbr.y_max + threshold - 1);
      intervals.emplace_back(mbr.y_min, mbr.y_max + threshold - 1);
    }
  }

  if (cell_ref_ids.empty())
    return {};

  /*since we discrete the y-axis coordinates, we preserve the coordinate->index
   * and index->coordinate mapping*/
  std::vector<int> index_to_coordinate(coordinates.begin(), coordinates.end());
  std::sort(index_to_coordinate.begin(), index_to_coordinate.end());

  std::vector<int> coordinate_to_index(index_to_coordinate.back() + 1);

  for (auto i = 0UL; i < index_to_coordinate.size(); ++i)
    coordinate_to_index[index_to_coordinate[i]] = i;

  // use disretized coordinate index to represent the intervals of cells
  for (auto& interval : intervals) {
    interval.first  = coordinate_to_index[interval.first];
    interval.second = coordinate_to_index[interval.second];
  }

  // merge intervals. end_points[i] is the max end point of the interval start
  // at i
  std::vector<int> end_points(coordinates.size());
  std::iota(end_points.begin(), end_points.end(), 0);

  for (const auto& interval : intervals) {
    auto l        = interval.first;
    auto r        = interval.second;
    end_points[l] = std::max(end_points[l], end_points[r]);
  }

  // get the partition result
  auto             row_id            = -1;
  auto             current_end_point = -1;
  std::vector<int> index_to_row(coordinates.size());
  for (auto i = 0UL; i < end_points.size(); ++i) {
    if (int(i) > current_end_point) {
      ++row_id;
    }
    index_to_row[i]   = row_id;
    current_end_point = std::max(current_end_point, end_points[i]);
  }

  // get the sub-rows
  std::vector<std::vector<int>> sub_rows(row_id + 1);
  for (auto idx = 0UL; idx < cell_ref_ids.size(); ++idx) {
    auto        cell_ref_id = cell_ref_ids[idx];
    const auto& interval    = intervals[idx];
    auto        point       = interval.second;
    sub_rows[index_to_row[point]].emplace_back(cell_ref_id);
  }
  return sub_rows;
}
}  // namespace odrc
