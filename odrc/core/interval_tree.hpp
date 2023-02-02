#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <set>
#include <vector>

namespace odrc::core {

template <typename T, typename V>
struct interval {
  T l;
  T r;
  V v;

  T    mid() const { return (l + r) / 2; }
  bool contains(const T& p) const { return l <= p and p <= r; }
};

template <typename T, typename V>
struct node {
  using interval_t = interval<T, V>;
  using Ovlp       = std::vector<std::pair<T, V>>;
  T    v;  // the value of the node is the mid point of the initial interval
  bool is_subtree_empty = false;

  node() = default;
  node(const interval_t& intvl) {
    v = intvl.mid();
    insert(intvl);
  }
  node(interval_t&& intvl) {
    v = intvl.mid();
    insert(std::move(intvl));
  }

  int lc = -1;  // left child offset;
  int rc = -1;  // right child offset

  struct l_compare {
    bool operator()(const interval_t& lhs, const interval_t& rhs) const {
      return lhs.l == rhs.l ? lhs.v < rhs.v : lhs.l < rhs.l;
    }
  };
  struct r_compare {
    bool operator()(const interval_t& lhs, const interval_t& rhs) const {
      return lhs.r == rhs.r ? lhs.v < rhs.v : lhs.r > rhs.r;
    }
  };

  std::set<interval_t, l_compare> intvls_l;
  // intervals in ascending left endpoint
  std::set<interval_t, r_compare> intvls_r;
  // intervals in descending right endpoint

  [[nodiscard]] bool has_left_child() const { return lc != -1; }
  [[nodiscard]] bool has_right_child() const { return rc != -1; }

  bool empty() const { return intvls_l.empty(); }

  void insert(const interval_t& intvl) {
    intvls_l.emplace(intvl);
    intvls_r.emplace(intvl);
    is_subtree_empty = false;
  }

  void insert(interval_t&& intvl) {
    intvls_l.emplace(std::move(intvl));
    intvls_r.emplace(std::move(intvl));
    is_subtree_empty = false;
  }

  void remove(const interval_t& intvl) {
    intvls_l.erase(intvl);
    intvls_r.erase(intvl);
  }

  void left_query(const T& p, const V& v, Ovlp& ovlp) const {
    for (auto it = intvls_l.begin(); it != intvls_l.end(); ++it) {
      if (it->l > p) {
        break;
      }
      ovlp.emplace_back(it->v, v);
    }
  }

  void right_query(const T& p, const V& v, Ovlp& ovlp) const {
    for (auto it = intvls_r.begin(); it != intvls_r.end(); ++it) {
      if (it->r < p) {
        break;
      }
      ovlp.emplace_back(std::make_pair(it->v, v));
    }
  }
};

template <typename T, typename V>
class interval_tree {
 public:
  using Intvl = interval<T, V>;
  using Node  = node<T, V>;
  using Ovlp  = std::vector<std::pair<T, V>>;
  void insert(const Intvl& intvl) {
    if (nodes.empty()) {
      nodes.emplace_back(intvl);
      return;
    }
    _insert(intvl, 0);
  };

  void remove(const Intvl& intvl, const std::size_t n = 0) {
    assert(n < nodes.size());
    Node& node = nodes.at(n);
    if (intvl.contains(node.v)) {
      node.remove(intvl);
    } else if (intvl.r < node.v) {
      remove(intvl, node.lc);
    } else {
      remove(intvl, node.rc);
    }
    bool is_left_empty =
        !node.has_left_child() or nodes.at(node.lc).is_subtree_empty;
    bool is_right_empty =
        !node.has_right_child() or nodes.at(node.rc).is_subtree_empty;
    node.is_subtree_empty = node.empty() and is_left_empty and is_right_empty;
  }

  void get_intervals_pairs(const Intvl& intvl, Ovlp& ovlps) {
    if (nodes.empty()) {
      return;
    }
    assert(intvl.l < intvl.r);
    _run_query(intvl, 0, ovlps);
  }

 private:
  std::vector<Node> nodes;
  void              _insert(const Intvl& intvl, const std::size_t n) {
    assert(n < nodes.size());
    Node& node            = nodes.at(n);
    node.is_subtree_empty = false;
    if (intvl.contains(node.v)) {
      node.insert(intvl);
    } else if (intvl.r < node.v) {  // insert to left subtree
      if (node.has_left_child()) {
        _insert(intvl, node.lc);
      } else {
        nodes.emplace_back(intvl);  // deactiavtes reference of node!
        nodes.at(n).lc = nodes.size() - 1;
      }
    } else {  // insert to right subtree
      if (node.has_right_child()) {
        _insert(intvl, node.rc);
      } else {
        nodes.emplace_back(intvl);  // deactiavtes reference of node!
        nodes.at(n).rc = nodes.size() - 1;
      }
    }
  };

  void _run_query(const Intvl& intvl, const std::size_t n, Ovlp& ovlps) {
    if (nodes.at(n).is_subtree_empty) {
      return;
    }
    assert(n < nodes.size());
    const Node& node = nodes.at(n);
    if (intvl.r <= node.v) {
      node.right_query(intvl.r, intvl.v, ovlps);
      if (node.has_left_child()) {
        _run_query(intvl, node.lc, ovlps);
      }
    }
    if (intvl.l >= node.v) {
      node.left_query(intvl.l, intvl.v, ovlps);
      if (node.has_right_child()) {
        _run_query(intvl, node.rc, ovlps);
      }
    }
  }
};

}  // namespace odrc::core