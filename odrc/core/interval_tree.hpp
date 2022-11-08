#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
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
  T mid;  // value of the tree node

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

  void insert(const Intvl& intvl) {
    intvl_l.emplace(intvl);
    intvl_r.emplace(intvl);
  }
  void insert(Intvl&& intvl) {
    intvl_l.emplace(std::move(intvl));
    intvl_r.emplace(std::move(intvl));
  }
  void remove(const Intvl& intvl) {
    intvl_l.erase(intvl);
    intvl_r.erase(intvl);
  }
  std::vector<V> get_intervals_containing(const T& p) const {
    std::vector<V> v;
    if (p <= mid) {  // iff intvl.l <= p
      for (auto it = intvl_l.begin(); it != intvl_l.end(); ++it) {
        if (it->l > p) {
          break;
        }
        v.emplace_back(it->v);
      }
    } else {
      for (auto it = intvl_r.begin(); it != intvl_r.end(); ++it) {
        if (it->r < p) {
          break;
        }
        v.emplace_back(it->v);
      }
    }
    return v;
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
    assert(n >= 0 and n < nodes.size());
    Node& node = nodes.at(n);
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
    assert(n >= 0 and n < nodes.size());
    Node& node = nodes.at(n);
    if (intvl.contains(node.mid)) {
      node.remove(intvl);
    } else if (intvl.r < node.mid) {
      remove(intvl, node.lc);
    } else {
      remove(intvl, node.rc);
    }
  }

  // returns a vector of Intvl::v
  std::vector<V> get_intervals_overlapping_with(const Intvl&      intvl,
                                                const std::size_t n = 0) {
    std::vector<V> intvls;
    if (nodes.empty()) {
      return intvls;
    }
    assert(n >= 0 and n < nodes.size());
    const Node& node = nodes.at(n);
    if (intvl.r <= node.mid) {
      auto v = node.get_intervals_containing(intvl.r);
      intvls.insert(intvls.end(), v.begin(), v.end());
      if (node.has_left_child()) {
        auto v_l = get_intervals_overlapping_with(intvl, node.lc);
        intvls.insert(intvls.end(), v_l.begin(), v_l.end());
      }
    } else if (intvl.l >= node.mid) {
      auto v = node.get_intervals_containing(intvl.l);
      intvls.insert(intvls.end(), v.begin(), v.end());
      if (node.has_right_child()) {
        auto v_r = get_intervals_overlapping_with(intvl, node.rc);
        intvls.insert(intvls.end(), v_r.begin(), v_r.end());
      }
    } else {
      auto v = node.get_intervals_containing(node.mid);
      intvls.insert(intvls.end(), v.begin(), v.end());
      if (node.has_left_child()) {
        auto v_l = get_intervals_overlapping_with(intvl, node.lc);
        intvls.insert(intvls.end(), v_l.begin(), v_l.end());
      }
      if (node.has_right_child()) {
        auto v_r = get_intervals_overlapping_with(intvl, node.rc);
        intvls.insert(intvls.end(), v_r.begin(), v_r.end());
      }
    }
  }

 private:
  std::vector<Node> nodes;
};

}  // namespace odrc::core