#pragma once

#include <odrc/core/database.hpp>

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <vector>

// give sub-rows, to indicate the cell_id and the corresponding cell_x,y

// layout_partition() declaration: partitions the whole cells in a specific
// layer into a set of sub-rows(sub-regions).
namespace odrc {
inline std::vector<std::vector<int>> layout_partition(odrc::core::database& db,
                                                      std::vector<int> layers) {
  const auto& cell_refs = db.cells.back().cell_refs;

  // get unique cell y-axis coordinates and store intervals need to merge
  std::unordered_set<int>          coordinates;
  std::vector<std::pair<int, int>> intervals;
  std::vector<int>                 cell_ids;

  for (auto id = 0UL; id < cell_refs.size(); ++id) {
    const auto& cell_ref = cell_refs[id];
    const auto& the_cell = db.get_cell(cell_ref.cell_name);
    if (the_cell.is_touching(layers.front())) {
      cell_ids.emplace_back(id);
      auto& mbr = db.edges[layers.front()].cell_ref_mbrs.at(id);
      coordinates.insert(mbr.y_min);
      coordinates.insert(mbr.y_max);
      intervals.emplace_back(mbr.y_min, mbr.y_max);
    } else if (the_cell.is_touching(layers.back())) {
      cell_ids.emplace_back(id);
      auto& mbr = db.edges[layers.back()].cell_ref_mbrs.at(id);
      coordinates.insert(mbr.y_min);
      coordinates.insert(mbr.y_max);
      intervals.emplace_back(mbr.y_min, mbr.y_max);
    } else {
      continue;
    }
  }
  if (cell_ids.empty())
    return {};
  /*since we discrete the y-axis coordinates, we preserve the coordinate->index
   * and index->coordinate mapping*/
  std::vector<int> index_to_coordinate(coordinates.begin(), coordinates.end());
  std::sort(index_to_coordinate.begin(), index_to_coordinate.end());

  std::vector<int> coordinate_to_index(index_to_coordinate.back() + 1);

  for (auto i = 0UL; i < index_to_coordinate.size(); ++i)
    coordinate_to_index[index_to_coordinate[i]] = i;

  // use index to represent the intervals of cells
  for (auto& interval : intervals) {
    interval.first  = coordinate_to_index[interval.first];
    interval.second = coordinate_to_index[interval.second];
  }

  // merge intervals.
  std::vector<int> rpoints(coordinates.size());
  std::iota(rpoints.begin(), rpoints.end(), 0);

  for (const auto& interval : intervals) {
    auto x     = interval.first;
    auto y     = interval.second;
    rpoints[x] = std::max(rpoints[x], rpoints[y]);
  }

  // get the partition result
  auto             row_id         = -1;
  auto             current_rpoint = -1;
  std::vector<int> coordinate_to_rows(coordinates.size());
  for (auto i = 0UL; i < rpoints.size(); ++i) {
    if (int(i) > current_rpoint) {
      ++row_id;
    }
    coordinate_to_rows[i] = row_id;
    current_rpoint        = std::max(current_rpoint, rpoints[i]);
  }

  // get the sub-rows
  std::vector<std::vector<int>> sub_rows(row_id + 1);
  for (auto id = 0UL; id < cell_ids.size(); ++id) {
    auto        cell_id  = cell_ids[id];
    const auto& interval = intervals[id];
    auto        rpoint   = interval.second;
    sub_rows[coordinate_to_rows[rpoint]].push_back(cell_id);
  }

  return sub_rows;
}
}  // namespace odrc
