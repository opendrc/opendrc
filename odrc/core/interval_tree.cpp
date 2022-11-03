#include <odrc/core/interval_tree.hpp>

#include <vector>

namespace odrc::core {

std::vector<std::pair<int, int>> interval_tree::get_intervals_containing_point(
    int point,
    int edge_id,
    int node) {
  std::vector<std::pair<int, int>> overlap_intervals;
  if (point == nodes.at(node).x_mid) {
    for (const auto& edge : nodes.at(node).lines_start) {
      overlap_intervals.emplace_back(std::make_pair(edge_id, edge.id));
    }
  } else if (point < nodes.at(node).x_mid) {
    for (auto edge = nodes.at(node).lines_start.begin();
         edge != nodes.at(node).lines_start.end(); edge++) {
      if (edge->x_left <= point) {
        overlap_intervals.emplace_back(std::make_pair(edge_id, edge->id));
      } else {
        break;
      }
    }
  } else if (point > nodes.at(node).x_mid) {
    for (auto edge = nodes.at(node).lines_end.rbegin();
         edge != (nodes.at(node).lines_end).rend(); edge++) {
      if (edge->x_right >= point) {
        overlap_intervals.emplace_back(std::make_pair(edge_id, edge->id));
      } else {
        break;
      }
    }
  }
  return overlap_intervals;
}

std::vector<std::pair<int, int>> interval_tree::overlap_interval_query(
    interval* edge,
    int       node) {
  std::vector<std::pair<int, int>> overlap_interval;
  if (edge->x_right <= nodes.at(node).x_mid) {
    auto overlap =
        get_intervals_containing_point(edge->x_right, edge->id, node);
    overlap_interval.insert(overlap_interval.end(), overlap.begin(),
                            overlap.end());
    if (nodes.at(node).left_son + 1) {
      auto overlap_cell = overlap_interval_query(edge, nodes.at(node).left_son);
      overlap_interval.insert(overlap_interval.end(), overlap_cell.begin(),
                              overlap_cell.end());
    }
  } else if (edge->x_left >= nodes.at(node).x_mid) {
    auto overlap = get_intervals_containing_point(edge->x_left, edge->id, node);
    overlap_interval.insert(overlap_interval.end(), overlap.begin(),
                            overlap.end());
    if (nodes.at(node).right_son + 1) {
      auto overlap_cell =
          overlap_interval_query(edge, nodes.at(node).right_son);
      overlap_interval.insert(overlap_interval.end(), overlap_cell.begin(),
                              overlap_cell.end());
    }
  } else {
    auto overlap =
        get_intervals_containing_point(nodes.at(node).x_mid, edge->id, node);
    overlap_interval.insert(overlap_interval.end(), overlap.begin(),
                            overlap.end());
    if (nodes.at(node).left_son + 1) {
      auto overlap_cell = overlap_interval_query(edge, nodes.at(node).left_son);
      overlap_interval.insert(overlap_interval.end(), overlap_cell.begin(),
                              overlap_cell.end());
    }
    if (nodes.at(node).right_son + 1) {
      auto overlap_cell = overlap_interval_query(edge, nodes.at(node).left_son);
      overlap_interval.insert(overlap_interval.end(), overlap_cell.begin(),
                              overlap_cell.end());
    }
  }
  return overlap_interval;
}

void interval_tree::add_segment(interval* edge, const int node) {
  auto start_insert = std::lower_bound(
      nodes.at(node).lines_start.begin(), nodes.at(node).lines_start.end(),
      edge->x_left, [](interval a, const int b) { return a.x_left < b; });
  nodes.at(node).lines_start.insert(start_insert, *edge);
  auto end_insert = std::lower_bound(
      nodes.at(node).lines_end.begin(), nodes.at(node).lines_end.end(),
      edge->x_right, [](interval a, const int b) { return a.x_right < b; });
  nodes.at(node).lines_end.insert(end_insert, *edge);
}

int interval_tree::add_node(interval* edge) {
  nodes.emplace_back();
  nodes.back().x_mid = (edge->x_left + edge->x_right) / 2;
  nodes.back().lines_start.emplace_back(*edge);
  nodes.back().lines_end.emplace_back(*edge);
  return nodes.size() - 1;
}

void interval_tree::add_interval(interval* edge, const int node) {
  if (edge->x_left > nodes.at(node).x_mid && (nodes.at(node).right_son + 1)) {
    add_interval(edge, nodes.at(node).right_son);
  } else if (edge->x_right < nodes.at(node).x_mid &&
             (nodes.at(node).left_son + 1)) {
    add_interval(edge, nodes.at(node).left_son);
  } else if (edge->x_left > nodes.at(node).x_mid &&
             (nodes.at(node).right_son + 1)) {
    nodes.at(node).right_son = add_node(edge);
  } else if (edge->x_right < nodes.at(node).x_mid &&
             (nodes.at(node).left_son + 1)) {
    nodes.at(node).left_son = add_node(edge);
  } else {
    add_segment(edge, node);
  }
}

void interval_tree::delete_interval(interval* edge, const int node) {
  if (edge->x_left > nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).right_son);
  } else if (edge->x_right < nodes.at(node).x_mid) {
    delete_interval(edge, nodes.at(node).left_son);
  } else {
    auto start_erase = std::lower_bound(
        nodes.at(node).lines_start.begin(), nodes.at(node).lines_start.end(),
        edge, [](interval a, interval* b) {
          return !(a.x_left < b->x_left && -a.id == b->id);
        });
    nodes.at(node).lines_start.erase(start_erase - 1);
    auto end_erase = std::lower_bound(
        nodes.at(node).lines_end.begin(), nodes.at(node).lines_end.end(), edge,
        [](interval a, interval* b) {
          return !(a.x_left < b->x_left && -a.id == b->id);
        });
    nodes.at(node).lines_end.erase(end_erase - 1);
  }
}
}  // namespace odrc::core