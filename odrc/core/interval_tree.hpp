#pragma once

#include <algorithm>
#include <vector>

namespace odrc::core {

bool is_exist(int child_ptr) {  // judge whether the child node is existing
  return child_ptr + 1;
}

struct interval {
  int x_left;
  int x_right;
  int y;
  int id;
};

struct node {
  int                   x_mid;
  int                   left_child  = -1;
  int                   right_child = -1;
  std::vector<interval> lines_start;
  std::vector<interval> lines_end;
};

class interval_tree {
 public:
  std::vector<node> nodes;

  int  add_node(interval* edge);  // add a node into the tree
  void add_interval(interval* edge,
                    const int node);  // add an interval into the interval tree
  void delete_interval(
      interval* edge,
      const int node);  // delete an interval from the interval tree
  void get_overlapping_intervals(
      interval* edge,
      const int node,
      std::vector<std::pair<int, int>>*
          overlap_interval);  // get the overlapping intervals

 private:
  void add_segment(interval* edge,
                   const int node);  // add a segment into an existing node
  void get_intervals_containing_point(
      const int point,
      const int edge_id,
      const int node,
      std::vector<std::pair<int, int>>*
          overlap_interval);  // get intervals containing the point
};

inline int interval_tree::add_node(interval* edge) {
  nodes.emplace_back();
  nodes.back().x_mid = (edge->x_left + edge->x_right) / 2;
  nodes.back().lines_start.emplace_back(*edge);
  nodes.back().lines_end.emplace_back(*edge);
  return nodes.size() - 1;
}

inline void interval_tree::get_intervals_containing_point(
    const int                         point,
    const int                         edge_id,
    const int                         node,
    std::vector<std::pair<int, int>>* overlap_intervals) {
  if (point == nodes.at(node).x_mid) {
    for (const auto& edge : nodes.at(node).lines_start) {
      overlap_intervals->emplace_back(std::make_pair(edge_id, edge.id));
    }
  } else if (point < nodes.at(node).x_mid) {
    for (auto edge = nodes.at(node).lines_start.begin();
         edge != nodes.at(node).lines_start.end(); edge++) {
      if (edge->x_left <= point) {
        overlap_intervals->emplace_back(std::make_pair(edge_id, edge->id));
      } else {
        break;
      }
    }
  } else if (point > nodes.at(node).x_mid) {
    for (auto edge = nodes.at(node).lines_end.rbegin();
         edge != nodes.at(node).lines_end.rend(); edge++) {
      if (edge->x_right >= point) {
        overlap_intervals->emplace_back(std::make_pair(edge_id, edge->id));
      } else {
        break;
      }
    }
  }
}

inline void interval_tree::get_overlapping_intervals(
    interval*                         edge,
    int                               node,
    std::vector<std::pair<int, int>>* overlap_interval) {
  int left_child_offset  = nodes.at(node).left_child;
  int right_child_offset = nodes.at(node).right_child;
  if (edge->x_right <= nodes.at(node).x_mid) {
    get_intervals_containing_point(edge->x_right, edge->id, node,
                                   overlap_interval);
    if (is_exist(left_child_offset)) {
      get_overlapping_intervals(edge, left_child_offset, overlap_interval);
    }

  } else if (edge->x_left >= nodes.at(node).x_mid) {
    get_intervals_containing_point(edge->x_left, edge->id, node,
                                   overlap_interval);
    if (is_exist(right_child_offset)) {
      get_overlapping_intervals(edge, right_child_offset, overlap_interval);
    }
  } else {
    get_intervals_containing_point(nodes.at(node).x_mid, edge->id, node,
                                   overlap_interval);
    if (is_exist(left_child_offset)) {
      get_overlapping_intervals(edge, left_child_offset, overlap_interval);
    }
    if (is_exist(right_child_offset)) {
      get_overlapping_intervals(edge, right_child_offset, overlap_interval);
    }
  }
}

inline void interval_tree::add_segment(interval* edge, const int node) {
  auto start_insert = std::lower_bound(
      nodes.at(node).lines_start.begin(), nodes.at(node).lines_start.end(),
      edge->x_left, [](interval a, const int b) { return a.x_left < b; });
  nodes.at(node).lines_start.insert(start_insert, *edge);
  auto end_insert = std::lower_bound(
      nodes.at(node).lines_end.begin(), nodes.at(node).lines_end.end(),
      edge->x_right, [](interval a, const int b) { return a.x_right < b; });
  nodes.at(node).lines_end.insert(end_insert, *edge);
}

inline void interval_tree::add_interval(interval* edge, const int node) {
  if (edge->x_left > nodes.at(node).x_mid) {
    if (is_exist(nodes.at(node).right_child)) {
      add_interval(edge, nodes.at(node).right_child);
    } else {
      nodes.at(node).right_child = add_node(edge);
    }
  } else if (edge->x_right < nodes.at(node).x_mid) {
    if (is_exist(nodes.at(node).left_child)) {
      add_interval(edge, nodes.at(node).left_child);
    } else {
      nodes.at(node).left_child = add_node(edge);
    }
  } else {
    add_segment(edge, node);
  }
}

inline void interval_tree::delete_interval(interval* edge, const int node) {
  if (edge->x_left > nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).right_child);
  } else if (edge->x_right < nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).left_child);
  } else {
    auto start_erase = std::lower_bound(
        nodes.at(node).lines_start.begin(), nodes.at(node).lines_start.end(),
        edge, [](interval a, interval* b) {
          return a.x_left < b->x_left && -a.id != b->id;
        });
    nodes.at(node).lines_start.erase(start_erase);
    auto end_erase = std::lower_bound(
        nodes.at(node).lines_end.begin(), nodes.at(node).lines_end.end(), edge,
        [](interval a, interval* b) {
          return a.x_right < b->x_right && -a.id != b->id;
        });
    nodes.at(node).lines_end.erase(end_erase);
  }
}
}  // namespace odrc::core