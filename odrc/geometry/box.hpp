#pragma once

#include <odrc/geometry/point.hpp>

namespace odrc::geometry {

template <typename Point = point2d<>>
class box {
 public:
  constexpr box() = default;
  constexpr box(const Point& min_corner, const Point& max_corner)
      : _min_corner(min_corner), _max_corner(max_corner) {}

  constexpr const Point& min_corner() const { return _min_corner; }
  constexpr const Point& max_corner() const { return _max_corner; }
  constexpr Point&       min_corner() { return _min_corner; }
  constexpr Point&       max_corner() { return _max_corner; }

  // Queries
  template <typename Geometry>
  bool overlaps(const Geometry& geo) = delete;

  bool overlaps(const box& other) {
    return _min_corner < other.max_corner() and
           _max_corner > other.min_corner();
  }

 private:
  Point _min_corner;
  Point _max_corner;
};

}  // namespace odrc::geometry