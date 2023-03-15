#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>

namespace odrc::geometry {

template <typename GeoSpace = geometry_space<>,
          bool IsClockWise  = true,
          template <typename T, typename Allocator> typename Container =
              std::vector,
          template <typename T> typename Allocator = std::allocator>
class polygon {
  static_assert(traits::is_valid_geometry_space_v<GeoSpace>);

 public:
  using vertex      = point<GeoSpace>;
  using vertex_list = Container<vertex, Allocator<vertex>>;

  polygon() = default;
  template <typename... Args>
  polygon(Args&&... args) : _vertices(std::forward<Args>(args)...) {}
  polygon(std::initializer_list<vertex> init) : _vertices(init) {}

  constexpr std::size_t size() const noexcept { return _vertices.size(); }

 private:
  vertex_list _vertices;
};

}  // namespace odrc::geometry