#pragma once

#include <climits>
#include <cstdint>
#include <iostream>
#include <map>
#include <odrc/core/edge.hpp>
#include <odrc/utility/datetime.hpp>
#include <string>
#include <vector>

namespace odrc::core {
struct envelope {
  int x_min = std::numeric_limits<int>::max();
  int x_max = std::numeric_limits<int>::min();
  int y_min = std::numeric_limits<int>::max();
  int y_max = std::numeric_limits<int>::min();
};

struct coord {
  int x;
  int y;
  coord(){};
  coord(int x, int y) {
    this->x = x;
    this->y = y;
  };

  coord(int x, int y, bool trans) {
    if (trans) {
      this->x = y;
      this->y = x;
    } else {
      this->x = x;
      this->y = y;
    }
  };
};

struct edge {
  coord point1;
  coord point2;
};

struct orthogonal_edge {
  int p_start;
  int p_end;
  int intercept;
};
}  // namespace odrc::core