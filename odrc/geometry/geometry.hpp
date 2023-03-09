#pragma once

#include <cstddef>
#include <type_traits>

namespace odrc::geometry {

struct cartesian {};

template <typename CoordinateType   = int,
          size_t Dimension          = 2,
          typename CoordinateSystem = cartesian>
struct geometry_space {
  using coordinate_type         = CoordinateType;
  using coordinate_system       = CoordinateSystem;
  static const size_t dimension = Dimension;
};

namespace traits {
template <typename GeoSpace,
          typename CoordinateType   = void,
          typename Dimension        = void,
          typename CoordinateSystem = void>
struct is_valid_geometry_space : std::false_type {};

template <typename GeoSpace>
struct is_valid_geometry_space<
    GeoSpace,
    std::void_t<typename GeoSpace::coordinate_type>,
    std::void_t<decltype(GeoSpace::dimension)>,
    std::void_t<typename GeoSpace::coordinate_system>> : std::true_type {};

template <typename GeoSpace>
inline constexpr bool is_valid_geometry_space_v =
    is_valid_geometry_space<GeoSpace>::value;
};  // namespace traits

}  // namespace odrc::geometry