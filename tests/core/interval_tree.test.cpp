#include <doctest/doctest.h>
#include <algorithm>
#include <cmath>
#include <odrc/core/interval_tree.hpp>
#include <odrc/gdsii/gdsii.hpp>
#include <vector>

using interval_pairs = std::vector<std::pair<int, int>>;

interval_pairs brute_force_query(std::vector<std::vector<int>>& mbr) {
  interval_pairs overlap_cells;
  for (int i = 0; i < mbr.size(); i++) {
    for (int j = i + 1; j < mbr.size(); j++) {
      bool is_overlapping =
          std::abs(mbr[i][1] + mbr[i][0] - mbr[j][1] - mbr[j][0]) <=
              mbr[i][1] + mbr[j][1] - mbr[i][0] - mbr[j][0] &&
          std::abs(mbr[i][3] + mbr[i][2] - mbr[j][3] - mbr[j][2]) <=
              mbr[i][3] + mbr[j][3] - mbr[i][2] - mbr[j][2];
      // Judge whether the center distance between two
      // rectangles is greater than half of the sum of
      // side lengths
      if (is_overlapping) {
        overlap_cells.emplace_back(std::make_pair(i + 1, j + 1));
      }
    }
  }
  return overlap_cells;
}

TEST_SUITE("[OpenDRC] odrc::core interval tree tests") {
  TEST_CASE("test overlapping cells") {
    std::vector<odrc::core::interval> sorted_edge;
    interval_pairs                    overlap_cells;
    // test data
    std::vector<std::vector<int>> mbr{
        {6, 12, 1, 4},  {3, 7, 2, 13},   {11, 15, 4, 14}, {2, 4, 5, 15},
        {6, 8, 6, 16},  {10, 12, 7, 17}, {14, 16, 8, 18}, {1, 3, 9, 19},
        {5, 13, 0, 12}, {3, 7, 3, 12}};  // the data represents
                                         // {x_min,x_max,y_min,y_max}
    for (int i = 0; i < mbr.size(); i++) {
      sorted_edge.emplace_back(odrc::core::interval{
          mbr[i][0], mbr[i][1], mbr[i][2],
          sorted_edge.size() / 2 + 1});  // input {x_min, x_max, y_min,id}
      sorted_edge.emplace_back(odrc::core::interval{
          mbr[i][0], mbr[i][1], mbr[i][3],
          -(sorted_edge.size() / 2 + 1)});  // input {x_min, x_max, y_max,-id}
    }

    std::sort(sorted_edge.begin(), sorted_edge.end(),
              [](const odrc::core::interval& a, const odrc::core::interval& b) {
                return a.y < b.y || (a.y == b.y && a.id > b.id);
              });  // sort edges by y (or id if y is equal)

    odrc::core::interval_tree tree;
    for (int edge = 0; edge != sorted_edge.size(); edge++) {
      if (sorted_edge[edge].id > 0) {
        // if y is greater than 0, query and add this edge
        auto overlap_intervals =
            tree.get_intervals_overlapping_with(sorted_edge[edge]);
        overlap_cells.insert(overlap_cells.end(), overlap_intervals.begin(),
                             overlap_intervals.end());
        tree.add_interval(sorted_edge[edge]);
      } else {
        // if y is less than 0, delete this edge
        sorted_edge[edge].id = std::abs(sorted_edge[edge].id);
        tree.delete_interval(sorted_edge[edge]);
      }
    }
    for (int i = 0; i != overlap_cells.size(); i++) {
      if (overlap_cells.at(i).first > overlap_cells.at(i).second) {
        std::swap(overlap_cells.at(i).first, overlap_cells.at(i).second);
      }
    }  // make sure the smaller id is the first element in pair
    std::sort(overlap_cells.begin(), overlap_cells.end());
    auto brute_force_query_result =
        brute_force_query(mbr);  // get brute force query result
    CHECK_EQ(overlap_cells.size(),
             brute_force_query_result.size());  // check the size of result
    for (int i = 0; i != brute_force_query_result.size(); i++) {
      CHECK_EQ(overlap_cells.at(i),
               brute_force_query_result.at(i));  // check every pair
    }
  }
}