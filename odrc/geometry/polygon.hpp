#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <odrc/geometry/point.hpp>

namespace odrc::geometry {

template <typename Point   = point2d<>,
          bool IsClockWise = true,
          template <typename T, typename Allocator> typename Container =
              std::vector,
          template <typename T> typename Allocator = std::allocator>
class polygon {
 public:
  using point_type = Point;
  using point_list = Container<Point, Allocator<Point>>;

  polygon() = default;
  polygon(const point_list& points) : _points(points) {}
  template <typename Iterator>
  polygon(Iterator begin, Iterator end) : _points(begin, end) {}
  polygon(std::initializer_list<Point> init) : _points(init) {}

  constexpr std::size_t size() const noexcept { return _points.size(); }

 private:
  point_list _points;
};
}  // namespace odrc::geometry