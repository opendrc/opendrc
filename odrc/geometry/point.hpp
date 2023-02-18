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

 private:
  CoordinateType values[Dimension];
};

template <typename CoordinateType = int>
class point2d : public point<CoordinateType, 2, cartesian> {
 public:
  constexpr point2d() = default;
  constexpr point2d(const CoordinateType& v0, const CoordinateType& v1)
      : point<CoordinateType, 2, cartesian>(v0, v1){};

  constexpr const CoordinateType& x() const { return this->template get<0>(); }
  constexpr const CoordinateType& y() const { return this->template get<1>(); }

  void x(const CoordinateType& v) { this->template set<0>(v); }
  void y(const CoordinateType& v) { this->template set<1>(v); }
};

}  // namespace odrc::geometry