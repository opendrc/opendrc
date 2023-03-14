#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include <odrc/geometry/box.hpp>

namespace odrc::geometry {

namespace traits {
template <typename Spatial>
struct spatial_traits {};
}  // namespace traits

template <typename Spatial>
class spatial {
  using geometry_space =
      typename traits::spatial_traits<Spatial>::geometry_space;
  using object_t = typename traits::spatial_traits<Spatial>::object_t;
  using box_t    = typename traits::spatial_traits<Spatial>::box_t;

 public:
  constexpr void insert(const object_t& obj) {
    static_cast<Spatial*>(this)->insert(obj);
  }
  template <typename OutIter>
  constexpr void query(const box_t& box, OutIter out_iter) {
    static_cast<Spatial*>(this)->query(box, out_iter);
  }
  constexpr std::size_t size() const noexcept {
    return static_cast<Spatial*>(this)->size();
  };

 private:
  constexpr spatial() = default;
  friend Spatial;
};

template <typename Object = box<geometry_space<>>>
class spatial_vector : public spatial<spatial_vector<Object>> {
  using geometry_space =
      typename traits::geo_space_traits<Object>::geometry_space;
  static_assert(traits::is_valid_geometry_space_v<geometry_space>);
  using object_t = Object;
  using box_t    = box<geometry_space>;

 public:
  template <typename... Args>
  spatial_vector(Args&&... args) : _objs(std::forward<Args>(args)...) {}
  spatial_vector(const std::initializer_list<Object>& init)
      : _objs(init.begin(), init.end()) {}
  constexpr void insert(const Object& obj) { _objs.emplace_back(obj); }

  template <typename OutIter>
  constexpr void query(const box_t& box, OutIter out_iter) {
    for (auto& obj : _objs) {
      if (obj.overlaps(box)) {
        *(out_iter++) = obj;
      }
    }
  }

  constexpr std::size_t size() const noexcept { return _objs.size(); }

 private:
  std::vector<Object> _objs;
};

namespace traits {
template <typename Object>
struct spatial_traits<spatial_vector<Object>> {
  using geometry_space =
      typename traits::geo_space_traits<Object>::geometry_space;
  using object_t = Object;
  using box_t    = box<geometry_space>;
};

}  // namespace traits

};  // namespace odrc::geometry