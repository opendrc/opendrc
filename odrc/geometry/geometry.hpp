#pragma once

#include <cstddef>
#include <type_traits>

namespace odrc::geometry {

struct cartesian {};

namespace traits {

template <typename T>
struct is_coordinate_system : std::false_type {};

template <>
struct is_coordinate_system<cartesian> : std::true_type {};

template <typename T>
inline constexpr bool is_coordinate_system_v = is_coordinate_system<T>::value;
}  // namespace traits

template <typename CoordinateType   = int,
          size_t Dimension          = 2,
          typename CoordinateSystem = cartesian>
struct geometry_space {
  using coordinate_type             = CoordinateType;
  using coordinate_system           = CoordinateSystem;
  static constexpr size_t dimension = Dimension;
};

namespace traits {

template <typename GeoSpace, typename = void>
struct geo_space_traits {};

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

template <typename GeoSpace, template <typename GeoSpaceT> typename Object>
struct geo_space_traits<Object<GeoSpace>,
                        std::enable_if_t<is_valid_geometry_space_v<GeoSpace>>> {
  using geometry_space = GeoSpace;
};
template <typename GeoSpace>
struct geo_space_traits<GeoSpace,
                        std::enable_if_t<is_valid_geometry_space_v<GeoSpace>>> {
  using geometry_space_t  = GeoSpace;
  using coordinate_type   = typename geometry_space_t::coordinate_type;
  using coordinate_system = typename geometry_space_t::coordinate_system;
  static constexpr size_t dimension = geometry_space_t::dimension;
};

}  // namespace traits

}  // namespace odrc::geometry