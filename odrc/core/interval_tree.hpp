#pragma once
#include <algorithm>
#include <vector>

namespace odrc::core {

using interval_pairs = std::vector<std::pair<int, int>>;

struct interval {
  int x_left;
  int x_right;
  int y;
  int id;
};

bool comp_left(const interval& a, const interval& b) {
 if (a.x_left == b.x_left) {
    return a.id < b.id;
  } else {
    return a.x_left < b.x_left;
  }
}

bool comp_right(const interval& a, const interval& b) {
  if (a.x_right == b.x_right) {
    return a.id < b.id;
  } else {
    return a.x_right < b.x_right;
  }
}

struct node {
  int                   x_mid;
  int                   left_child  = -1;
  int                   right_child = -1;
  std::vector<interval> lines_start;
  std::vector<interval> lines_end;

  bool has_left_child() { return left_child != -1; }
  bool has_right_child() { return right_child != -1; }
};

class interval_tree {
 public:
  void           add_interval(const interval& edge, const int node = 0);
  void           delete_interval(const interval& edge, const int node = 0);
  interval_pairs get_intervals_overlapping_with(const interval& edge,
                                                const int       node = 0);

 private:
  std::vector<node> nodes;

  int  _add_node(const interval& edge);
  void _add_interval_to_node(
      const interval& edge,
      const int       node);  // add a segment into an existing node
  interval_pairs _get_intervals_containing_point(
      const int point,
      const int edge_id,
      const int node);  // get intervals containing the point
};

inline int interval_tree::_add_node(const interval& edge) {
  nodes.emplace_back();
  nodes.back().x_mid = (edge.x_left + edge.x_right) / 2;
  nodes.back().lines_start.emplace_back(edge);
  nodes.back().lines_end.emplace_back(edge);
  return nodes.size() - 1;
}

inline interval_pairs interval_tree::_get_intervals_containing_point(
    const int point,
    const int id,
    const int node) {
  interval_pairs overlap_intervals;
  if (point <= nodes.at(node).x_mid) {
    for (auto edge = nodes.at(node).lines_start.begin();
         edge != nodes.at(node).lines_start.end(); edge++) {
      if (edge->x_left <= point) {
        overlap_intervals.emplace_back(std::make_pair(id, edge->id));
      } else {
        break;
      }
    }
  } else if (point > nodes.at(node).x_mid) {
    for (auto edge = nodes.at(node).lines_end.rbegin();
         edge != nodes.at(node).lines_end.rend(); edge++) {
      if (edge->x_right >= point) {
        overlap_intervals.emplace_back(std::make_pair(id, edge->id));
      } else {
        break;
      }
    }
  }
  return overlap_intervals;
}

inline interval_pairs interval_tree::get_intervals_overlapping_with(
    const interval& edge,
    const int       node) {
  interval_pairs overlap_intervals;
  if (nodes.empty()) {  // if node is empty, return null
    return overlap_intervals;
  }
  int left_child_offset  = nodes.at(node).left_child;
  int right_child_offset = nodes.at(node).right_child;
  if (edge.x_right <= nodes.at(node).x_mid) {
    overlap_intervals =
        _get_intervals_containing_point(edge.x_right, edge.id, node);
    if (nodes.at(node).has_left_child()) {  // if node has a left child, jump
                                            // to left child node
      auto intervals = get_intervals_overlapping_with(edge, left_child_offset);
      overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                               intervals.end());
    }
  } else if (edge.x_left >= nodes.at(node).x_mid) {
    overlap_intervals =
        _get_intervals_containing_point(edge.x_left, edge.id, node);
    if (nodes.at(node).has_right_child()) {
      auto intervals = get_intervals_overlapping_with(edge, right_child_offset);
      overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                               intervals.end());
    }
  } else {
    overlap_intervals =
        _get_intervals_containing_point(nodes.at(node).x_mid, edge.id, node);
    if (nodes.at(node).has_left_child()) {
      auto intervals = get_intervals_overlapping_with(edge, left_child_offset);
      overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                               intervals.end());
    }
    if (nodes.at(node).has_right_child()) {
      auto intervals = get_intervals_overlapping_with(edge, right_child_offset);
      overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                               intervals.end());
    }
  }
  return overlap_intervals;
}

inline void interval_tree::_add_interval_to_node(const interval& edge,
                                                 const int       node) {
  auto start_insert =
      std::lower_bound(nodes.at(node).lines_start.begin(),
                       nodes.at(node).lines_start.end(), edge, comp_left);
  nodes.at(node).lines_start.insert(start_insert, edge);
  auto end_insert =
      std::lower_bound(nodes.at(node).lines_end.begin(),
                       nodes.at(node).lines_end.end(), edge, comp_right);
  nodes.at(node).lines_end.insert(end_insert, edge);
}

inline void interval_tree::add_interval(const interval& edge, const int node) {
  if (nodes.empty()) {
    _add_node(edge);
    return;
  }
  if (edge.x_left > nodes.at(node).x_mid) {
    if (nodes.at(node).has_right_child()) {
      add_interval(edge, nodes.at(node).right_child);
    } else {
      nodes.at(node).right_child = _add_node(edge);
    }
  } else if (edge.x_right < nodes.at(node).x_mid) {
    if (nodes.at(node).has_left_child()) {
      add_interval(edge, nodes.at(node).left_child);
    } else {
      nodes.at(node).left_child = _add_node(edge);
    }
  } else {
    _add_interval_to_node(edge, node);
  }
}

inline void interval_tree::delete_interval(const interval& edge,
                                           const int       node) {
  if (edge.x_left > nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).right_child);
  } else if (edge.x_right < nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).left_child);
  } else {
    auto start_erase =
        std::lower_bound(nodes.at(node).lines_start.begin(),
                         nodes.at(node).lines_start.end(), edge, comp_left);
    nodes.at(node).lines_start.erase(start_erase);
    auto end_erase =
        std::lower_bound(nodes.at(node).lines_end.begin(),
                         nodes.at(node).lines_end.end(), edge, comp_right);
    nodes.at(node).lines_end.erase(end_erase);
  }
}
}  // namespace odrc::core