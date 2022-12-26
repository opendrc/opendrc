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
  using Intvl = interval<T, V>;
  T    mid;  // value of the tree node
  bool is_subtree_empty = false;

  node() = default;
  node(const Intvl& intvl) {
    mid = intvl.mid();
    insert(intvl);
  }
  node(Intvl&& intvl) {
    mid = intvl.mid();
    insert(std::move(intvl));
  }

  int lc = -1;  // left child offset;
  int rc = -1;  // right child offset

  using l_comp = struct {
    bool operator()(const Intvl& lhs, const Intvl& rhs) const {
      return lhs.l == rhs.l ? lhs.v < rhs.v : lhs.l < rhs.l;
    }
  };
  using r_comp = struct {
    bool operator()(const Intvl& lhs, const Intvl& rhs) const {
      return lhs.r == rhs.r ? lhs.v < rhs.v : lhs.r > rhs.r;
    }
  };

  std::set<Intvl, l_comp> intvl_l;  // intervals in ascending left endpoint
  std::set<Intvl, r_comp> intvl_r;  // intervals in descending right endpoint

  [[nodiscard]] bool has_left_child() const { return lc != -1; }
  [[nodiscard]] bool has_right_child() const { return rc != -1; }

  bool empty() const { return intvl_l.empty(); }

  void insert(const Intvl& intvl) {
    intvl_l.emplace(intvl);
    intvl_r.emplace(intvl);
    is_subtree_empty = false;
  }
  void insert(Intvl&& intvl) {
    intvl_l.emplace(std::move(intvl));
    intvl_r.emplace(std::move(intvl));
    is_subtree_empty = false;
  }
  void remove(const Intvl& intvl) {
    intvl_l.erase(intvl);
    intvl_r.erase(intvl);
  }
  void get_intervals_containing(const T&                      p,
                                const V&                      v,
                                std::vector<std::pair<V, V>>& ovlp,
                                bool                          reverse) const {
    if (p <= mid) {
      for (auto it = intvl_l.begin(); it != intvl_l.end(); ++it) {
        if (it->l > p) {
          break;
        }
        if (reverse) {
          ovlp.emplace_back(std::make_pair(it->v, v));
        } else {
          ovlp.emplace_back(std::make_pair(v, it->v));
        }
      }
    } else {
      for (auto it = intvl_r.begin(); it != intvl_r.end(); ++it) {
        if (it->r < p) {
          break;
        }
        if (reverse) {
          ovlp.emplace_back(std::make_pair(it->v, v));
        } else {
          ovlp.emplace_back(std::make_pair(v, it->v));
        }
      }
    }
  }
};

template <typename T, typename V>
class interval_tree {
 public:
  using Intvl = interval<T, V>;
  using Node  = node<T, V>;

  void insert(const Intvl& intvl, const std::size_t n = 0) {
    if (nodes.empty()) {
      nodes.emplace_back(intvl);
      return;
    }
    assert(n < nodes.size());
    Node& node            = nodes.at(n);
    node.is_subtree_empty = false;
    if (intvl.contains(node.mid)) {
      node.insert(intvl);
    } else if (intvl.r < node.mid) {  // insert to left subtree
      if (node.has_left_child()) {
        insert(intvl, node.lc);
      } else {
        nodes.emplace_back(intvl);  // deactiavtes reference of node!
        nodes.at(n).lc = nodes.size() - 1;
      }
    } else {  // insert to right subtree
      if (node.has_right_child()) {
        insert(intvl, node.rc);
      } else {
        nodes.emplace_back(intvl);  // deactiavtes reference of node!
        nodes.at(n).rc = nodes.size() - 1;
      }
    }
  };

  void remove(const Intvl& intvl, const std::size_t n = 0) {
    if (n < nodes.size()) {
    } else {
      std::cout << n << std::endl;
      std::cout << intvl.l << " " << intvl.r << " " << intvl.mid() << " "
                << intvl.v << std::endl;
    }
    assert(n < nodes.size());
    Node& node = nodes.at(n);
    if (intvl.contains(node.mid)) {
      node.remove(intvl);
    } else if (intvl.r < node.mid) {
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
  void get_intervals_overlapping_with(const Intvl&                  intvl,
                                      std::vector<std::pair<T, V>>& ovlp) {
    _run_query(intvl, 0, ovlp);
  }

  // returns a vector of Intvl::v
  void _run_query(const Intvl&                  intvl,
                  const std::size_t             n,
                  std::vector<std::pair<T, V>>& rtn) {
    if (nodes.empty()) {
      return;
    }
    if (nodes.at(n).is_subtree_empty) {
      return;
    }
    assert(n < nodes.size());
    const Node& node = nodes.at(n);
    if (intvl.r <= node.mid) {
      node.get_intervals_containing(intvl.r, intvl.v, rtn, reverse);
      if (node.has_left_child()) {
        _run_query(intvl, node.lc, rtn);
      }
    } else if (intvl.l >= node.mid) {
      node.get_intervals_containing(intvl.l, intvl.v, rtn, reverse);
      if (node.has_right_child()) {
        _run_query(intvl, node.rc, rtn);
      }
    } else {
      node.get_intervals_containing(node.mid, intvl.v, rtn, reverse);
      if (node.has_left_child()) {
        _run_query(intvl, node.lc, rtn);
      }
      if (node.has_right_child()) {
        _run_query(intvl, node.rc, rtn);
      }
    }
  }
  bool reverse = false;

 private:
  std::vector<Node> nodes;
};

}  // namespace odrc::core