#if false
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
    using Intvl = odrc::core::interval<int, int>;
    struct edge {
      Intvl intvl;
      int   y;
      bool  is_remove;
      edge(int l, int r, int y, int id, bool is_remove = false)
          : intvl(Intvl{l, r, id}), y(y), is_remove(is_remove) {}
    };
    std::vector<edge> edges;
    interval_pairs    overlap_cells;
    // test data
    std::vector<std::vector<int>> mbr{
        {6, 12, 1, 4}, {3, 7, 2, 13},   {11, 15, 4, 14}, {2, 4, 5, 15},
        {6, 8, 6, 16}, {10, 12, 7, 17}, {14, 16, 8, 18}, {1, 3, 9, 19},
        {5, 13, 0, 12}};  // the data represents {x_min,x_max,y_min,y_max}
    for (int i = 0; i < 9; ++i) {
      edges.emplace_back(mbr[i][0], mbr[i][1], mbr[i][2], i);
      edges.emplace_back(mbr[i][0], mbr[i][1], mbr[i][3], i, true);
    }

    std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
      return a.y == b.y ? b.is_remove  : a.y < b.y;
    });  // sort edges by y (or id if y is equal)
    std::cout<<"edges size: "<<edges.size()<<std::endl;
    odrc::core::interval_tree<int, int> tree;
    for (const auto& edge : edges) {
      if (!edge.is_remove) {
        // auto overlap_intervals =
            // tree.get_intervals_pairs(edge.intvl);
        // for (int e : overlap_intervals) {
        //   overlap_cells.emplace_back(edge.intvl.v+1, e+1);
        // }
        tree.insert(edge.intvl);
      } else {
        tree.remove(edge.intvl);
      }
    }
    for (auto& cells : overlap_cells) {
      if (cells.first > cells.second) {
        std::swap(cells.first, cells.second);
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
#endif