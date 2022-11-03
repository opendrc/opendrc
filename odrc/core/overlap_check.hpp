#include <list>
#include <map>
#include <vector>
namespace odrc::core {

struct mbr_edge {
  int x_left;
  int x_right;
  int y;
  int id;
};

struct line {
  int x_left;
  int x_right;
  int line_id;
};

struct node {
  int               x_mid;
  bool              is_left    = false;
  bool              is_right   = false;
  int               left_son   = -1;
  int               right_son  = -1;
  int               parent_ptr = -1;
  std::vector<line> lines_start;
  std::vector<line> lines_end;
};

class interval_tree {
 public:
  std::vector<node>                nodes;
  std::vector<std::pair<int, int>> inter_cells;
  int                              add_node(const int&, const int&, const int&);
};

void add_interval(const int&,
                  const int&,
                  const int&,
                  const int&,
                  interval_tree&);
void add_line(const int&, const int&, const int&, const int&, interval_tree&);
void delete_interval(const int&,
                     const int&,
                     const int&,
                     const int&,
                     interval_tree&);
void interval_check(const int&,
                   const int&,
                   const int&,
                   const int&,
                   interval_tree&);
void point_check(const bool&,
                 const int&,
                 const int&,
                 const int&,
                 interval_tree&);
void pushup(const int&, interval_tree&);

void overlap_check(odrc::core::database& db, const int& layer) {
  std::vector<mbr_edge> sorted;

  std::vector<int> x_discrete;
  std::vector<int> y_discrete;

  int id = 1;
  // input data
  for (const auto& cel : db.cells) {
    for (const auto& pol : cel.polygons) {
      if (pol.layer == layer) {
        sorted.emplace_back(mbr_edge{pol.mbr[0], pol.mbr[1], pol.mbr[2], id});
        sorted.emplace_back(mbr_edge{pol.mbr[0], pol.mbr[1], pol.mbr[3], -id});
        x_discrete.emplace_back(pol.mbr[0]);
        x_discrete.emplace_back(pol.mbr[1]);
        y_discrete.emplace_back(pol.mbr[2]);
        y_discrete.emplace_back(pol.mbr[3]);
        id++;
      }
    }
  }

  // discrete data
  std::sort(x_discrete.begin(), x_discrete.end());
  x_discrete.erase(std::unique(x_discrete.begin(), x_discrete.end()),
                   x_discrete.end());
  std::sort(y_discrete.begin(), y_discrete.end());
  y_discrete.erase(std::unique(y_discrete.begin(), y_discrete.end()),
                   y_discrete.end());

  for (auto sor = sorted.begin(); sor != sorted.end(); sor++) {
    sor->x_left =
        std::lower_bound(x_discrete.begin(), x_discrete.end(), sor->x_left) -
        x_discrete.begin() + 1;
    sor->x_right =
        std::lower_bound(x_discrete.begin(), x_discrete.end(), sor->x_right) -
        x_discrete.begin() + 1;
    sor->y = std::lower_bound(y_discrete.begin(), y_discrete.end(), sor->y) -
             y_discrete.begin() + 1;
  }
  // // Your data
  // int mbr[9][4] = {{5, 13, 0, 12},  {6, 12, 1, 4},   {3, 7, 2, 13},
  //                  {11, 15, 4, 14}, {2, 4, 5, 15},   {6, 8, 6, 16},
  //                  {10, 12, 7, 17}, {14, 16, 8, 18}, {1, 3, 9, 19}};
  // for (int i = 0; i < 9; i++) {
  //   sorted.emplace_back(mbr_edge{mbr[i][0], mbr[i][1], mbr[i][2], id});
  //   sorted.emplace_back(mbr_edge{mbr[i][0], mbr[i][1], mbr[i][3], -id});
  //   id++;
  // }

  std::sort(sorted.begin(), sorted.end(),
            [](const mbr_edge& a, const mbr_edge& b) { return a.y < b.y; });
  interval_tree tree;
  tree.add_node(sorted.at(0).x_left, sorted.at(0).x_right,
                sorted.at(0).id);  // the first node

  for (auto sor = sorted.begin() + 1; sor != sorted.end(); sor++) {
    if (sor->id > 0) {
      interval_check(sor->x_left, sor->x_right, sor->id, 0, tree);
      add_interval(sor->x_left, sor->x_right, sor->id, 0, tree);
    } else {
      delete_interval(sor->x_left, sor->x_right, -sor->id, 0, tree);
    }
  }
}

void add_line(const int&     x_left,
              const int&     x_right,
              const int&     line_id,
              const int&     node,
              interval_tree& tree) {
  if (tree.nodes.at(node).lines_start.size() == 0) {
    tree.nodes.at(node).lines_start.emplace_back(
        line{x_left, x_right, line_id});
    tree.nodes.at(node).lines_end.emplace_back(line{x_left, x_right, line_id});
  } else {
    auto start_insert = std::lower_bound(
        tree.nodes.at(node).lines_start.begin(),
        tree.nodes.at(node).lines_start.end(), x_left,
        [](const line& a, const int b) { return a.x_left < b; });
    tree.nodes.at(node).lines_start.insert(start_insert,
                                           line{x_left, x_right, line_id});
    auto end_insert = std::lower_bound(
        tree.nodes.at(node).lines_end.begin(),
        tree.nodes.at(node).lines_end.end(), x_right,
        [](const line a, const int b) { return a.x_right < b; });
    tree.nodes.at(node).lines_end.insert(end_insert,
                                         line{x_left, x_right, line_id});
  }
}

int interval_tree::add_node(const int& x_left,
                            const int& x_right,
                            const int& line_id) {
  nodes.emplace_back();
  nodes.back().x_mid = (x_left + x_right) / 2;
  if (nodes.back().lines_start.size() == 0) {
    nodes.back().lines_start.emplace_back(line{x_left, x_right, line_id});
    nodes.back().lines_end.emplace_back(line{x_left, x_right, line_id});
  } else {
    auto start_insert = std::lower_bound(
        nodes.back().lines_start.begin(), nodes.back().lines_start.end(),
        x_left, [](const line& a, const int b) { return a.x_left < b; });
    nodes.back().lines_start.insert(start_insert,
                                    line{x_left, x_right, line_id});
    auto end_insert = std::lower_bound(
        nodes.back().lines_end.begin(), nodes.back().lines_end.end(), x_right,
        [](const line a, const int b) { return a.x_right < b; });
    nodes.back().lines_end.insert(end_insert, line{x_left, x_right, line_id});
  }
  return (nodes.size() - 1);
}

void add_interval(const int&     x_left,
                  const int&     x_right,
                  const int&     line_id,
                  const int&     node,
                  interval_tree& tree) {
  if (x_left > tree.nodes.at(node).x_mid && tree.nodes.at(node).is_right) {
    add_interval(x_left, x_right, line_id, tree.nodes.at(node).right_son, tree);
  } else if (x_right < tree.nodes.at(node).x_mid &&
             tree.nodes.at(node).is_left) {
    add_interval(x_left, x_right, line_id, tree.nodes.at(node).left_son, tree);
  } else if (x_left > tree.nodes.at(node).x_mid &&
             tree.nodes.at(node).is_right == false) {
    tree.nodes.at(node).is_right  = true;
    tree.nodes.at(node).right_son = tree.add_node(x_left, x_right, line_id);
    tree.nodes.back().parent_ptr  = node;
  } else if (x_right < tree.nodes.at(node).x_mid &&
             tree.nodes.at(node).is_left == false) {
    tree.nodes.at(node).is_left  = true;
    tree.nodes.at(node).left_son = tree.add_node(x_left, x_right, line_id);
    tree.nodes.back().parent_ptr = node;
  } else {
    add_line(x_left, x_right, line_id, node, tree);
  }
}

void point_check(const bool&    is_cover,
                 const int&     point,
                 const int&     node,
                 const int&     line_id,
                 interval_tree& tree) {
  if (is_cover || point == tree.nodes.at(node).x_mid) {
    for (auto&& line : tree.nodes.at(node).lines_start) {
      tree.inter_cells.emplace_back(std::make_pair(line_id, line.line_id));
    }
  } else if (point < tree.nodes.at(node).x_mid) {
    for (auto line = tree.nodes.at(node).lines_start.begin();
         line != tree.nodes.at(node).lines_start.end(); line++) {
      if (line->x_left < point) {
        tree.inter_cells.emplace_back(std::make_pair(line_id, line->line_id));
      } else {
        break;
      }
    }
  } else if (point > tree.nodes.at(node).x_mid) {
    for (auto line = tree.nodes.at(node).lines_end.rbegin();
         line != (tree.nodes.at(node).lines_end).rend(); line++) {
      if (line->x_right > point) {
        tree.inter_cells.emplace_back(std::make_pair(line_id, line->line_id));
      } else {
        break;
      }
    }
  }
}

void pushup(const int& node, interval_tree& tree) {
  int parent_ptr = tree.nodes.at(node).parent_ptr;
  if (parent_ptr != -1) {
    if (tree.nodes.at(node).lines_start.size() == 0 &&
        tree.nodes.at(node).lines_end.size() == 0 &&
        tree.nodes.at(node).is_left == false &&
        tree.nodes.at(node).is_right == false) {
      if (tree.nodes.at(node).x_mid > tree.nodes.at(parent_ptr).x_mid) {
        tree.nodes.at(parent_ptr).right_son = -1;
        tree.nodes.at(parent_ptr).is_right  = false;
      } else {
        tree.nodes.at(parent_ptr).left_son = -1;
        tree.nodes.at(parent_ptr).is_left  = false;
      }
    }
    pushup(parent_ptr, tree);
  }
}

void interval_check(const int&     x_left,
                   const int&     x_right,
                   const int&     line_id,
                   const int&     node,
                   interval_tree& tree) {
  if (x_right <= tree.nodes.at(node).x_mid) {
    point_check(false, x_right, node, line_id, tree);
    if (tree.nodes.at(node).is_left) {
      interval_check(x_left, x_right, line_id, tree.nodes.at(node).left_son,
                    tree);
    }
  } else if (x_left >= tree.nodes.at(node).x_mid) {
    point_check(false, x_left, node, line_id, tree);
    if (tree.nodes.at(node).is_right) {
      interval_check(x_left, x_right, line_id, tree.nodes.at(node).right_son,
                    tree);
    }
  } else {
    point_check(true, x_right, node, line_id, tree);
    if (tree.nodes.at(node).is_left) {
      interval_check(x_left, x_right, line_id, tree.nodes.at(node).left_son,
                    tree);
    }
    if (tree.nodes.at(node).is_right) {
      interval_check(x_left, x_right, line_id, tree.nodes.at(node).right_son,
                    tree);
    }
  }
}

void delete_interval(const int&     x_left,
                     const int&     x_right,
                     const int&     line_id,
                     const int&     node,
                     interval_tree& tree) {
  if (x_left > tree.nodes.at(node).x_mid) {
    delete_interval(x_left, x_right, line_id, tree.nodes.at(node).right_son,
                    tree);
  } else if (x_right < tree.nodes.at(node).x_mid) {
    delete_interval(x_left, x_right, line_id, tree.nodes.at(node).left_son,
                    tree);
  } else {
    auto lin = tree.nodes.at(node).lines_start.begin();
    while (true) {
      if (lin->line_id == line_id) {
        tree.nodes.at(node).lines_start.erase(lin);
        break;
      } else {
        lin++;
      }
    }
    auto line = tree.nodes.at(node).lines_end.begin();
    while (true) {
      if (line->line_id == line_id) {
        tree.nodes.at(node).lines_end.erase(line);
        break;
      } else {
        line++;
      }
    }
    pushup(node, tree);
  }
}

}  // namespace odrc::core