#include <odrc/utility/proxy.hpp>

#include <list>
#include <vector>

#include <doctest/doctest.h>

TEST_SUITE("[OpenDRC] odrc::util proxy tests") {
  TEST_CASE("test const_container_proxy for vector") {
    std::vector<int>            v{2, 3, 5, 7};
    odrc::util::const_container_proxy vp(v);
    CHECK_EQ(vp.begin(), v.begin());
    CHECK_EQ(vp.end(), v.end());
    std::vector<int> v2;
    for (int i : vp) {
      v2.emplace_back(i);
    }
    CHECK_EQ(v.size(), v2.size());
    CHECK_EQ(v, v2);
  }
  TEST_CASE("test const_container_proxy for list") {
    std::list<int>              l{2, 3, 5, 7};
    odrc::util::const_container_proxy lp(l);
    CHECK_EQ(lp.begin(), l.begin());
    CHECK_EQ(lp.end(), l.end());
    std::list<int> l2;
    for (int i : lp) {
      l2.emplace_back(i);
    }
    CHECK_EQ(l.size(), l2.size());
    CHECK_EQ(l, l2);
  }
}