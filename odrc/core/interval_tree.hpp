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
  using intvl_t    = interval<T, V>;
  using ovlp_intvl = std::vector<V>;
  T    key;  // the value of the node is the mid point of the initial interval
  bool is_subtree_empty = false;

  node() = default;
  node(const intvl_t& intvl) {
    key = intvl.mid();
    insert(intvl);
  }
  node(intvl_t&& intvl) {
    key = intvl.mid();
    insert(std::move(intvl));
  }

  int lc = -1;  // left child offset;
  int rc = -1;  // right child offset

  struct l_compare {
    bool operator()(const intvl_t& lhs, const intvl_t& rhs) const {
      return lhs.l == rhs.l ? lhs.v < rhs.v : lhs.l < rhs.l;
    }
  };
  struct r_compare {
    bool operator()(const intvl_t& lhs, const intvl_t& rhs) const {
      return lhs.r == rhs.r ? lhs.v < rhs.v : lhs.r > rhs.r;
    }
  };

  std::set<intvl_t, l_compare> intvls_l;
  // intervals in ascending left endpoint
  std::set<intvl_t, r_compare> intvls_r;
  // intervals in descending right endpoint

  [[nodiscard]] bool has_left_child() const { return lc != -1; }
  [[nodiscard]] bool has_right_child() const { return rc != -1; }

  bool empty() const { return intvls_l.empty(); }

  void insert(const intvl_t& intvl) {
    intvls_l.emplace(intvl);
    intvls_r.emplace(intvl);
    is_subtree_empty = false;
  }

  void insert(intvl_t&& intvl) {
    intvls_l.emplace(std::move(intvl));
    intvls_r.emplace(std::move(intvl));
    is_subtree_empty = false;
  }

  void remove(const intvl_t& intvl) {
    intvls_l.erase(intvl);
    intvls_r.erase(intvl);
  }

  void query(const T& p, ovlp_intvl& ovlp) const {
    if (p < this->key) {
      for (auto it = intvls_l.begin(); it != intvls_l.end(); ++it) {
        if (it->l > p) {
          break;
        }
        ovlp.emplace_back(it->v);
      }
    } else {
      for (auto it = intvls_r.begin(); it != intvls_r.end(); ++it) {
        if (it->r < p) {
          break;
        }
        ovlp.emplace_back(it->v);
      }
    }
  }
};

template <typename T, typename V>
class interval_tree {
 public:
  using intvl_t    = interval<T, V>;
  using node_t     = node<T, V>;
  using ovlp_intvl = std::vector<V>;
  void insert(const intvl_t& intvl) {
    if (nodes.empty()) {
      nodes.emplace_back(intvl);
      return;
    }
    _insert(intvl, 0);
  };

  void remove(const intvl_t& intvl) { _remove(intvl, 0); }

  ovlp_intvl query(const intvl_t& intvl) {
    ovlp_intvl ovlp_intvls;
    if (!nodes.empty()) {
      assert(intvl.l < intvl.r);
      _intvl_query(intvl, 0, ovlp_intvls);
    }
    return ovlp_intvls;
  }

  ovlp_intvl query(const T& p) const {
    ovlp_intvl ovlp_intvls;
    if (!nodes.empty()) {
      _point_query(p, 0, ovlp_intvls);
    }
    return ovlp_intvls;
  }

 private:
  std::vector<node_t> nodes;
  void                update_state(node_t& node) {
    bool is_left_empty =
        !node.has_left_child() or nodes.at(node.lc).is_subtree_empty;
    bool is_right_empty =
        !node.has_right_child() or nodes.at(node.rc).is_subtree_empty;
    node.is_subtree_empty = node.empty() and is_left_empty and is_right_empty;
  }

  void _insert(const intvl_t& intvl, const std::size_t n) {
    assert(n < nodes.size());
    node_t& node          = nodes.at(n);
    node.is_subtree_empty = false;
    if (intvl.contains(node.key)) {
      node.insert(intvl);
    } else if (intvl.r < node.key) {  // insert to left subtree
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
  }

  void _remove(const intvl_t& intvl, const std::size_t n) {
    assert(n < nodes.size());
    node_t& node = nodes.at(n);
    if (intvl.contains(node.key)) {
      node.remove(intvl);
    } else if (intvl.r < node.key) {
      _remove(intvl, node.lc);
    } else {
      _remove(intvl, node.rc);
    }
    update_state(node);
  }

  void _intvl_query(const intvl_t&    intvl,
                    const std::size_t n,
                    ovlp_intvl&       ovlps) {
    if (nodes.at(n).is_subtree_empty) {
      return;
    }
    assert(n < nodes.size());
    const node_t& node = nodes.at(n);
    if (intvl.r < node.key) {
      node.query(intvl.r, ovlps);
      if (node.has_left_child()) {
        _intvl_query(intvl, node.lc, ovlps);
      }
    } else if (intvl.l >= node.key) {
      node.query(intvl.l, ovlps);
      if (node.has_right_child()) {
        _intvl_query(intvl, node.rc, ovlps);
      }
    } else {
      node.query(node.key, ovlps);
      if (node.has_left_child()) {
        _intvl_query(intvl, node.lc, ovlps);
      }
      if (node.has_right_child()) {
        _intvl_query(intvl, node.rc, ovlps);
      }
    }
  }

  void _point_query(const T& p, const std::size_t n, ovlp_intvl& ovlps) {
    if (nodes.at(n).is_subtree_empty) {
      return;
    }
    assert(n < nodes.size());
    const node_t& node = nodes.at(n);
    node.query(p, ovlps);
    if (p < node.key and node.has_left_child()) {
      _point_query(p, node.lc, ovlps);
    } else if (p >= node.key and node.has_right_child()) {
      _point_query(p, node.rc, ovlps);
    }
  }
};

}  // namespace odrc::core