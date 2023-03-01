#pragma once

#include <odrc/geometry/polygon.hpp>

namespace odrc::db {
template <typename Polygon = odrc::geometry::polygon<>>
class element {
 public:
  using polygon_type = Polygon;

  constexpr element() = default;
  constexpr element(const Polygon& poly) : _poly(poly) {}
  constexpr element(Polygon&& poly) : _poly(std::move(poly)) {}

  constexpr Polygon&       get_polygon() { return _poly; }
  constexpr const Polygon& get_polygon() const { return _poly; }

 private:
  Polygon _poly;
};
}  // namespace odrc::db