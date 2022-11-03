#pragma once

#include <vector>

namespace odrc::core {

struct interval {
  int x_left;
  int x_right;
  int y;
  int id;
};

struct node {
  int                   x_mid;
  int                   left_son   = -1;
  int                   right_son  = -1;
  int                   parent_ptr = -1;
  std::vector<interval> lines_start;
  std::vector<interval> lines_end;
};

class interval_tree {
 public:
  std::vector<node>                nodes;
  int                              add_node(interval*);
  std::vector<std::pair<int, int>> overlap_interval_query(interval*, const int);
  void                             add_interval(interval*, const int);
  void                             delete_interval(interval*, const int);

 private:
  void                             add_segment(interval*, const int);
  std::vector<std::pair<int, int>> get_intervals_containing_point(int,
                                                                  int,
                                                                  int);
};

}  // namespace odrc::core