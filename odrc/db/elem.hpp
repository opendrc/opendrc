#pragma once

#include <odrc/geometry/polygon.hpp>

namespace odrc::db {
template <typename Polygon = odrc::geometry::polygon<>>
class element {
  using geo_space = typename odrc::geometry::traits::geo_space_traits<
      Polygon>::geometry_space;
  using point_t = odrc::geometry::point<geo_space>;

 public:
  using polygon_t     = Polygon;
  constexpr element() = default;
  constexpr element(int layer, const Polygon& poly)
      : _layer(layer), _poly(poly) {}
  constexpr element(int layer, Polygon&& poly)
      : _layer(layer), _poly(std::move(poly)) {}

  // attributes

  constexpr int            get_layer() const noexcept { return _layer; }
  constexpr Polygon&       get_polygon() noexcept { return _poly; }
  constexpr const Polygon& get_polygon() const noexcept { return _poly; }

  void set_layer(int layer) noexcept { _layer = layer; }

  // operations

  constexpr element operator+(const point_t& point) const {
    return element(_layer, _poly + point);
  }

 private:
  int     _layer;
  Polygon _poly;
};

}  // namespace odrc::db

namespace odrc::geometry::traits {
template <typename Polygon>
struct geo_space_traits<odrc::db::element<Polygon>> {
  using geometry_space = typename geo_space_traits<Polygon>::geometry_space;
};
}  // namespace odrc::geometry::traits