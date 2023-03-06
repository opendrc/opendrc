#pragma once

#include <algorithm>
#include <cstddef>

#include <odrc/geometry/coordinate_system.hpp>

namespace odrc::geometry {
template <typename CoordinateType   = int,
          std::size_t Dimension     = 2,
          typename CoordinateSystem = cartesian>
class point {
  static_assert(Dimension > 0, "Dimension of a point must be positive.");

 public:
  constexpr point() = default;
  constexpr point(const CoordinateType& v0, const CoordinateType& v1)
      : values{v0, v1} {}
  constexpr point(std::initializer_list<CoordinateType> init) {
    std::copy(init.begin(), init.end(), values);
  }

  template <std::size_t K>
  constexpr const CoordinateType& get() const {
    static_assert(K < Dimension);
    return values[K];
  }

  template <std::size_t K>
  void set(const CoordinateType& v) {
    static_assert(K < Dimension);
    values[K] = v;
  }

  bool operator<(const point& other) const {
    for (std::size_t i = 0; i < Dimension; ++i) {
      if (values[i] >= other.values[i])
        return false;
    }
    return true;
  }
  bool operator>(const point& other) const { return other < *this; }
  bool operator<=(const point& other) const {
    for (std::size_t i = 0; i < Dimension; ++i) {
      if (values[i] > other.values[i])
        return false;
    }
    return true;
  }
  bool operator>=(const point& other) const { return other <= *this; }
  bool operator==(const point& other) const {
    return std::equal(values, values + Dimension, other.values,
                      other.values + Dimension);
  }
  bool operator!=(const point& other) const { return !(*this == other); }

 private:
  CoordinateType values[Dimension];
};

template <typename CoordinateType = int>
class point2d : public point<CoordinateType, 2, cartesian> {
 public:
  constexpr point2d() = default;
  constexpr point2d(const CoordinateType& v0, const CoordinateType& v1)
      : point<CoordinateType>(v0, v1) {}
  constexpr point2d(std::initializer_list<CoordinateType> init)
      : point<CoordinateType>{init} {}

  constexpr const CoordinateType& x() const { return this->template get<0>(); }
  constexpr const CoordinateType& y() const { return this->template get<1>(); }

  void x(const CoordinateType& v) { this->template set<0>(v); }
  void y(const CoordinateType& v) { this->template set<1>(v); }
};

}  // namespace odrc::geometry