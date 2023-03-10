#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>

namespace odrc::geometry {

template <typename GeoSpace = geometry_space<>>
class polygon {
 public:
  template <typename Polygon>
  polygon(const Polygon& polygon)
      : _this(std::make_shared<model<Polygon>>(polygon)) {}

  std::size_t size() const { return _this->size(); }

 private:
  class concept {
   public:
    virtual ~concept()                        = default;
    virtual std::size_t size() const noexcept = 0;
  };
  template <typename Polygon>

  class model : public concept {
   public:
    model(const Polygon& polygon) : _polygon(polygon) {}
    model(Polygon&& polygon) : _polygon(std::move(polygon)) {}
    std::size_t size() const noexcept override { return _polygon.size(); }

   private:
    Polygon _polygon;
  };

  std::shared_ptr<concept> _this;
};

namespace detail {

template <typename GeoSpace = geometry_space<>,
          bool IsClockWise  = true,
          template <typename T, typename Allocator> typename Container =
              std::vector,
          template <typename T> typename Allocator = std::allocator>
class polygon {
 public:
  using vertex      = point<GeoSpace>;
  using vertex_list = Container<vertex, Allocator<vertex>>;

  polygon() = default;
  polygon(const vertex_list& vertices) : _vertices(vertices) {}
  template <typename Iterator>
  polygon(Iterator begin, Iterator end) : _vertices(begin, end) {}
  polygon(std::initializer_list<vertex> init) : _vertices(init) {}

  constexpr std::size_t size() const noexcept { return _vertices.size(); }

 private:
  vertex_list _vertices;
};

}  // namespace detail
}  // namespace odrc::geometry