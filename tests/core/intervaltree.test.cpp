#include <vector>

#include <doctest/doctest.h>

#include <odrc/core/interval_tree.hpp>
#include <odrc/gdsii/gdsii.hpp>

TEST_SUITE("[OpenDRC] odrc::core interval tree tests") {
  TEST_CASE("test overlapping cells") {
    std::vector<odrc::core::interval> sorted_edge;
    std::vector<std::pair<int, int>>  overlap_cells;

    // Your data
    int mbr[9][4] = {{5, 13, 0, 12},  {6, 12, 1, 4},   {3, 7, 2, 13},
                     {11, 15, 4, 14}, {2, 4, 5, 15},   {6, 8, 6, 16},
                     {10, 12, 7, 17}, {14, 16, 8, 18}, {1, 3, 9, 19}};
    for (int i = 0; i < 9; i++) {
      sorted_edge.emplace_back(
          odrc::core::interval{mbr[i][0], mbr[i][1], mbr[i][2], sorted_edge.size() / 2 + 1});
      sorted_edge.emplace_back(
          odrc::core::interval{mbr[i][0], mbr[i][1], mbr[i][3], -sorted_edge.size() / 2 -1});
    }
    std::sort(sorted_edge.begin(), sorted_edge.end(),
              [](const odrc::core::interval& a, const odrc::core::interval& b) {
                return a.y < b.y;
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
    CHECK_EQ(overlap_cells.at(0).first,2);
    CHECK_EQ(overlap_cells.at(0).second,1);
    CHECK_EQ(overlap_cells.at(1).first,3);
    CHECK_EQ(overlap_cells.at(1).second,1);
    CHECK_EQ(overlap_cells.at(2).first,3);
    CHECK_EQ(overlap_cells.at(2).second,2);
    CHECK_EQ(overlap_cells.at(3).first,4);
    CHECK_EQ(overlap_cells.at(3).second,1);
    CHECK_EQ(overlap_cells.at(4).first,5);
    CHECK_EQ(overlap_cells.at(4).second,3);
    CHECK_EQ(overlap_cells.at(5).first,6);
    CHECK_EQ(overlap_cells.at(5).second,1);
    CHECK_EQ(overlap_cells.at(6).first,6);
    CHECK_EQ(overlap_cells.at(6).second,3);
    CHECK_EQ(overlap_cells.at(7).first,7);
    CHECK_EQ(overlap_cells.at(7).second,1);
    CHECK_EQ(overlap_cells.at(8).first,7);
    CHECK_EQ(overlap_cells.at(8).second,4);
    CHECK_EQ(overlap_cells.at(9).first,8);
    CHECK_EQ(overlap_cells.at(9).second,4);
    CHECK_EQ(overlap_cells.at(10).first,9);
    CHECK_EQ(overlap_cells.at(10).second,3);
    CHECK_EQ(overlap_cells.at(11).first,9);
    CHECK_EQ(overlap_cells.at(11).second,5);
  }
}