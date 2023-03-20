#pragma once

#include <odrc/db/cell.hpp>
#include <odrc/geometry/spatial.hpp>

namespace odrc::db {
template <typename Cell = cell<>,
          typename Spatial =
              odrc::geometry::spatial_vector<typename Cell::element_t>>
class layer {
  using element_t =
      typename odrc::geometry::traits::spatial_traits<Spatial>::object_t;

 public:
  constexpr layer(int l) noexcept : _layer_id(l) {}

  void add(const element_t& elem) { _spatial.insert(elem); }

  auto size() const noexcept { return _spatial.size(); }

 private:
  int     _layer_id;
  Spatial _spatial;
};
}  // namespace odrc::db