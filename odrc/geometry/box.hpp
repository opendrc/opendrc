#pragma once

#include <any>
#include <cstddef>
#include <iostream>
#include <memory>

#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>

namespace odrc::geometry {

template <typename GeoSpace = geometry_space<>>
class box {
  static_assert(traits::is_valid_geometry_space_v<GeoSpace>);
  using point_t = point<GeoSpace>;

 public:
  constexpr box() = default;
  constexpr box(const point_t& min_corner, const point_t& max_corner)
      : _min_corner(min_corner), _max_corner(max_corner) {}

  constexpr const point_t& min_corner() const { return _min_corner; }
  constexpr const point_t& max_corner() const { return _max_corner; }
  constexpr point_t&       min_corner() { return _min_corner; }
  constexpr point_t&       max_corner() { return _max_corner; }

  constexpr bool operator==(const box& other) const noexcept {
    return _min_corner == other._min_corner and
           _max_corner == other._max_corner;
  }

  // Queries
  template <typename Geometry>
  bool overlaps(const Geometry& geo) = delete;

  bool overlaps(const box& other) {
    return _min_corner < other.max_corner() and
           _max_corner > other.min_corner();
  }

 private:
  point_t _min_corner;
  point_t _max_corner;
};
}  // namespace odrc::geometry