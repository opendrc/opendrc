#include <doctest/doctest.h>
#include <algorithm>
#include <cmath>
#include <odrc/core/interval_tree.hpp>
#include <odrc/gdsii/gdsii.hpp>
#include <vector>

using interval_pairs = std::vector<std::pair<int, int>>;

bool comp(const std::pair<int, int>& a, const std::pair<int, int>& b) {
  if (a.first < b.first) {
    return true;
  } else if (a.first == b.first) {
    return a.second < b.second;
  } else {
    return false;
  }
}

interval_pairs brute_force_query(std::vector<std::vector<int>>& mbr) {
  interval_pairs overlap_cells;
  for (int i = 0; i < mbr.size(); i++) {
    for (int j = i + 1; j < mbr.size(); j++) {
      bool is_overlapping =
          std::abs(mbr[i][1] + mbr[i][0] - mbr[j][1] - mbr[j][0]) <=
              mbr[i][1] + mbr[j][1] - mbr[i][0] - mbr[j][0] &&
          std::abs(mbr[i][3] + mbr[i][2] - mbr[j][3] - mbr[j][2]) <=
              mbr[i][3] + mbr[j][3] - mbr[i][2] - mbr[j][2];
      if (is_overlapping) {
        overlap_cells.emplace_back(std::make_pair(i + 1, j + 1));
      }
    }
  }
  std::sort(overlap_cells.begin(), overlap_cells.end(), comp);
  return overlap_cells;
}

TEST_SUITE("[OpenDRC] odrc::core interval tree tests") {
  TEST_CASE("test overlapping cells") {
    std::vector<odrc::core::interval> sorted_edge;
    interval_pairs                    overlap_cells;
    // test data
    std::vector<std::vector<int>> mbr{
        {6, 12, 1, 4},   {3, 7, 2, 13}, {11, 15, 4, 14},
        {2, 4, 5, 15},   {6, 8, 6, 16}, {10, 12, 7, 17},
        {14, 16, 8, 18}, {1, 3, 9, 19}, {5, 13, 0, 12}};
    for (int i = 0; i < 9; i++) {
      sorted_edge.emplace_back(odrc::core::interval{
          mbr[i][0], mbr[i][1], mbr[i][2], sorted_edge.size() / 2 + 1});
      sorted_edge.emplace_back(odrc::core::interval{
          mbr[i][0], mbr[i][1], mbr[i][3], -sorted_edge.size() / 2 - 1});
    }

    std::sort(sorted_edge.begin(), sorted_edge.end(),
              [](const odrc::core::interval& a, const odrc::core::interval& b) {
                return a.y < b.y || (a.y == b.y && a.id > b.id);
              });
              
    odrc::core::interval_tree tree;
    for (int edge = 0; edge != sorted_edge.size(); edge++) {
      if (sorted_edge[edge].id > 0) {
        auto overlap_intervals =
            tree.get_intervals_overlapping_with(sorted_edge[edge]);
        overlap_cells.insert(overlap_cells.end(), overlap_intervals.begin(),
                             overlap_intervals.end());
        tree.add_interval(sorted_edge[edge]);
      } else {
        tree.delete_interval(sorted_edge[edge]);
      }
    }
    for (int i = 0; i != overlap_cells.size(); i++) {
      if (overlap_cells.at(i).first > overlap_cells.at(i).second) {
        std::pair<int, int> pair_swap{overlap_cells.at(i).second,
                                      overlap_cells.at(i).first};
        overlap_cells.at(i).swap(pair_swap);
      }
    }
    std::sort(overlap_cells.begin(), overlap_cells.end(), comp);
    auto brute_force_query_result = brute_force_query(mbr);
    CHECK_EQ(overlap_cells.size(), brute_force_query_result.size());
    for (int i = 0; i != brute_force_query_result.size(); i++) {
      CHECK_EQ(overlap_cells.at(i).first, brute_force_query_result.at(i).first);
      CHECK_EQ(overlap_cells.at(i).second,
               brute_force_query_result.at(i).second);
    }
  }
}