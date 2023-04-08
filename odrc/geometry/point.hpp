#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

#include <odrc/geometry/geometry.hpp>

namespace odrc::geometry {

template <typename GeoSpace = geometry_space<>>
class point {
  static_assert(traits::is_valid_geometry_space_v<GeoSpace>);
  using coordinate_type =
      typename traits::geo_space_traits<GeoSpace>::coordinate_type;
  static constexpr size_t dimension =
      traits::geo_space_traits<GeoSpace>::dimension;

 public:
  constexpr point() = default;
  constexpr point(const coordinate_type& v0, const coordinate_type& v1)
      : _values{v0, v1} {}
  constexpr point(std::initializer_list<coordinate_type> init) {
    std::copy(init.begin(), init.end(), _values.begin());
  }

  constexpr coordinate_type& operator[](size_t pos) { return _values[pos]; }
  constexpr const coordinate_type& operator[](size_t pos) const {
    return _values[pos];
  }

  constexpr coordinate_type& at(size_t pos) { return _values.at(pos); }

  constexpr const coordinate_type& at(size_t pos) const {
    return _values.at(pos);
  }

  constexpr coordinate_type&       x() { return at(0); }
  constexpr coordinate_type&       y() { return at(1); }
  constexpr coordinate_type&       z() { return at(2); }
  constexpr coordinate_type&       m() { return at(3); }
  constexpr const coordinate_type& x() const { return at(0); }
  constexpr const coordinate_type& y() const { return at(1); }
  constexpr const coordinate_type& z() const { return at(2); }
  constexpr const coordinate_type& m() const { return at(3); }

  constexpr void x(const coordinate_type& v) { at(0) = v; }
  constexpr void y(const coordinate_type& v) { at(1) = v; }
  constexpr void z(const coordinate_type& v) { at(2) = v; }
  constexpr void m(const coordinate_type& v) { at(3) = v; }

  constexpr bool operator<(const point& other) const {
    for (std::size_t i = 0; i < dimension; ++i) {
      if (_values[i] >= other._values[i])
        return false;
    }
    return true;
  }
  constexpr bool operator>(const point& other) const { return other < *this; }
  constexpr bool operator<=(const point& other) const {
    for (std::size_t i = 0; i < dimension; ++i) {
      if (_values[i] > other._values[i]) {
        return false;
      }
    }
    return true;
  }
  constexpr bool operator>=(const point& other) const { return other <= *this; }
  constexpr bool operator==(const point& other) const {
    return _values == other._values;
  }
  constexpr bool operator!=(const point& other) const {
    return !(*this == other);
  }

  constexpr point operator+(const point& other) const {
    point result;
    std::transform(_values.begin(), _values.end(), other._values.begin(),
                   result._values.begin(), std::plus<coordinate_type>());
    return result;
  }

 private:
  std::array<coordinate_type, dimension> _values{};
};

}  // namespace odrc::geometry