#pragma once

namespace odrc::geometry {
struct coordinate_system {};

struct cartesian {};
struct cartesian_tag {};

namespace traits {
template <typename CoordinateSystem>
struct cs_tag {};

template <>
struct cs_tag<odrc::geometry::cartesian> {
  using type = cartesian_tag;
};

}  // namespace traits
}  // namespace odrc::geometry