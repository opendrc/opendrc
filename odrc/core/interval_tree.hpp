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
  void add_interval(
      interval& edge,
      const int node = 0);  // add an interval into the interval tree
  void delete_interval(
      interval& edge,
      const int node = 0);  // delete an interval from the interval tree
  interval_pairs get_intervals_overlapping_with(
      interval& edge,
      const int node = 0);  // get the intervals overlapping with the given edge

 private:
  std::vector<node> nodes;

  int  _add_node(interval& edge);
  void _add_interval_to_node(
      interval& edge,
      const int node);  // add a segment into an existing node
  interval_pairs _get_intervals_containing_point(
      const int point,
      const int edge_id,
      const int node);  // get intervals containing the point
};

inline int interval_tree::_add_node(interval& edge) {
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
  if (point ==
      nodes.at(node).x_mid) {  // if the point is the middle of the node ,then
                               // all intervals will contain this point
    for (const auto& edge : nodes.at(node).lines_end) {
      overlap_intervals.emplace_back(std::make_pair(id, edge.id));
    }
  } else if (point < nodes.at(node).x_mid) {
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
    interval& edge,
    int       node) {
  interval_pairs overlap_intervals;
  if (nodes.empty()) {  // if node is empty, return null
    return overlap_intervals;
  } else {
    int left_child_offset  = nodes.at(node).left_child;
    int right_child_offset = nodes.at(node).right_child;
    if (edge.x_right <= nodes.at(node).x_mid) {
      overlap_intervals =
          _get_intervals_containing_point(edge.x_right, edge.id, node);
      if (nodes.at(node).has_left_child()) {  // if node has a left child, jump
                                              // to left child node
        auto intervals =
            get_intervals_overlapping_with(edge, left_child_offset);
        overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                                 intervals.end());
      }
    } else if (edge.x_left >= nodes.at(node).x_mid) {
      overlap_intervals =
          _get_intervals_containing_point(edge.x_left, edge.id, node);
      if (nodes.at(node).has_right_child()) {
        auto intervals =
            get_intervals_overlapping_with(edge, right_child_offset);
        overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                                 intervals.end());
      }
    } else {
      overlap_intervals =
          _get_intervals_containing_point(nodes.at(node).x_mid, edge.id, node);
      if (nodes.at(node).has_left_child()) {
        auto intervals =
            get_intervals_overlapping_with(edge, left_child_offset);
        overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                                 intervals.end());
      }
      if (nodes.at(node).has_right_child()) {
        auto intervals =
            get_intervals_overlapping_with(edge, right_child_offset);
        overlap_intervals.insert(overlap_intervals.end(), intervals.begin(),
                                 intervals.end());
      }
    }
    return overlap_intervals;
  }
}

inline void interval_tree::_add_interval_to_node(interval& edge,
                                                 const int node) {
  auto start_insert = std::lower_bound(
      nodes.at(node).lines_start.begin(), nodes.at(node).lines_start.end(),
      edge.x_left, [](interval a, const int b) { return a.x_left < b; });
  nodes.at(node).lines_start.insert(start_insert, edge);
  auto end_insert = std::lower_bound(
      nodes.at(node).lines_end.begin(), nodes.at(node).lines_end.end(),
      edge.x_right, [](interval a, const int b) { return a.x_right < b; });
  nodes.at(node).lines_end.insert(end_insert, edge);
}

inline void interval_tree::add_interval(interval& edge, int node) {
  if (nodes.empty()) {
    _add_node(edge);
  } else {
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
}

inline void interval_tree::delete_interval(interval& edge, const int node) {
  if (edge.x_left > nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).right_child);
  } else if (edge.x_right < nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).left_child);
  } else {
    auto start_erase = std::lower_bound(
        nodes.at(node).lines_start.begin(), nodes.at(node).lines_start.end(),
        edge, [](interval a, interval b) {
          return a.x_left < b.x_left && -a.id != b.id;
        });
    nodes.at(node).lines_start.erase(start_erase);
    auto end_erase = std::lower_bound(
        nodes.at(node).lines_end.begin(), nodes.at(node).lines_end.end(), edge,
        [](interval a, interval b) {
          return a.x_right < b.x_right && -a.id != b.id;
        });
    nodes.at(node).lines_end.erase(end_erase);
  }
}
}  // namespace odrc::core