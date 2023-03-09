#pragma once

#include <any>
#include <cstddef>
#include <memory>

#include <odrc/geometry/geometry.hpp>
#include <odrc/geometry/point.hpp>

namespace odrc::geometry {

template <typename GeoSpace>
class box {
  using point_t = point<GeoSpace>;

 public:
  virtual const point_t& min_corner() const = 0;
  virtual const point_t& max_corner() const = 0;
  virtual point_t&       min_corner()       = 0;
  virtual point_t&       max_corner()       = 0;

  // Queries
  virtual bool overlaps(const std::any& geo) const = 0;
  virtual bool overlaps(const box& other) const    = 0;

 private:
  struct concept {
   public:
    virtual ~concept() = default;

    virtual const point_t& min_corner() const = 0;
    virtual const point_t& max_corner() const = 0;
    virtual point_t&       min_corner()       = 0;
    virtual point_t&       max_corner()       = 0;

    // Queries
    virtual bool overlaps(const std::any& geo) const  = 0;
    virtual bool overlaps(const concept& other) const = 0;
  };
  template <typename Box>
  struct model : public concept {
   public:
    model(const Box& box) : _box(box) {}
    model(Box&& box) : _box(std::move(box)) {}

    const point_t& min_corner() const override { return _box.min_corner(); }
    const point_t& max_corner() const override { return _box.max_corner(); }
    point_t&       min_corner() override { return _box.min_corner(); }
    point_t&       max_corner() override { return _box.max_corner(); }

    bool overlaps(const std::any& geo) const override = delete;
    bool overlaps(const box& other) const override {
      return _box.overlaps(other);
    }

   private:
    Box _box;
  };

 private:
  std::shared_ptr<concept> _box;
};

template <typename GeoSpace>
class boxnd {
  using point_t = point<GeoSpace>;

 public:
  constexpr boxnd() = default;
  constexpr boxnd(const point_t& min_corner, const point_t& max_corner)
      : _min_corner(min_corner), _max_corner(max_corner) {}

  constexpr const point_t& min_corner() const { return _min_corner; }
  constexpr const point_t& max_corner() const { return _max_corner; }
  constexpr point_t&       min_corner() { return _min_corner; }
  constexpr point_t&       max_corner() { return _max_corner; }

  // Queries
  template <typename Geometry>
  bool overlaps(const Geometry& geo) = delete;

  bool overlaps(const boxnd& other) {
    return _min_corner < other.max_corner() and
           _max_corner > other.min_corner();
  }

 private:
  point_t _min_corner;
  point_t _max_corner;
};
}  // namespace odrc::geometry