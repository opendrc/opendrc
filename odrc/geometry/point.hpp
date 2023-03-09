#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <odrc/geometry/geometry.hpp>

namespace odrc::geometry {

template <typename GeoSpace = geometry_space<>>
class point {
  static_assert(traits::is_valid_geometry_space_v<GeoSpace>);

 public:
  using coordinate_type = typename GeoSpace::coordinate_type;
  template <typename Point>
  point(const Point& p) : _this(std::make_shared<model<Point>>(p)) {}

  coordinate_type&       x() { return _this->x(); }
  coordinate_type&       y() { return _this->y(); }
  coordinate_type&       z() { return _this->z(); }
  coordinate_type&       m() { return _this->m(); }
  const coordinate_type& x() const { return _this->x(); }
  const coordinate_type& y() const { return _this->y(); }
  const coordinate_type& z() const { return _this->z(); }
  const coordinate_type& m() const { return _this->m(); }

  void x(const coordinate_type& v) { _this->x(v); }
  void y(const coordinate_type& v) { _this->y(v); }
  void z(const coordinate_type& v) { _this->z(v); }
  void m(const coordinate_type& v) { _this->m(v); }

  bool operator<(const point& other) const { return *_this < (*other._this); };
  bool operator>(const point& other) const { return *_this > (*other._this); };
  bool operator<=(const point& other) const {
    return *_this <= (*other._this);
  };
  bool operator>=(const point& other) const {
    return *_this >= (*other._this);
  };
  bool operator==(const point& other) const {
    return *_this == (*other._this);
  };
  bool operator!=(const point& other) const {
    return *_this != (*other._this);
  };

 private:
  struct concept {
   public:
    virtual ~concept() = default;

    virtual coordinate_type&       x()       = 0;
    virtual coordinate_type&       y()       = 0;
    virtual coordinate_type&       z()       = 0;
    virtual coordinate_type&       m()       = 0;
    virtual const coordinate_type& x() const = 0;
    virtual const coordinate_type& y() const = 0;
    virtual const coordinate_type& z() const = 0;
    virtual const coordinate_type& m() const = 0;

    virtual void x(const coordinate_type& v) = 0;
    virtual void y(const coordinate_type& v) = 0;
    virtual void z(const coordinate_type& v) = 0;
    virtual void m(const coordinate_type& v) = 0;

    virtual bool operator<(const concept& other) const  = 0;
    virtual bool operator>(const concept& other) const  = 0;
    virtual bool operator<=(const concept& other) const = 0;
    virtual bool operator>=(const concept& other) const = 0;
    virtual bool operator==(const concept& other) const = 0;
    virtual bool operator!=(const concept& other) const = 0;
  };
  template <typename Point>
  struct model : public concept {
   public:
    model(const Point& p) : _p(p) {}

    coordinate_type&       x() override { return _p.x(); }
    coordinate_type&       y() override { return _p.y(); }
    coordinate_type&       z() override { return _p.z(); }
    coordinate_type&       m() override { return _p.m(); }
    const coordinate_type& x() const override { return _p.x(); };
    const coordinate_type& y() const override { return _p.y(); };
    const coordinate_type& z() const override { return _p.z(); };
    const coordinate_type& m() const override { return _p.m(); };

    void x(const coordinate_type& v) override { return _p.x(v); }
    void y(const coordinate_type& v) override { return _p.y(v); }
    void z(const coordinate_type& v) override { return _p.z(v); }
    void m(const coordinate_type& v) override { return _p.m(v); }

    bool operator<(const concept& other) const override {
      return _p < dynamic_cast<const model&>(other)._p;
    }
    bool operator>(const concept& other) const override {
      return _p > dynamic_cast<const model&>(other)._p;
    }
    bool operator<=(const concept& other) const override {
      return _p <= dynamic_cast<const model&>(other)._p;
    }
    bool operator>=(const concept& other) const override {
      return _p >= dynamic_cast<const model&>(other)._p;
    }
    bool operator==(const concept& other) const override {
      return _p == dynamic_cast<const model&>(other)._p;
    }
    bool operator!=(const concept& other) const override {
      return _p != dynamic_cast<const model&>(other)._p;
    }

   private:
    Point _p;
  };

  std::shared_ptr<concept> _this;
};

template <typename GeoSpace = geometry_space<>>
class pointnd {
  static_assert(traits::is_valid_geometry_space_v<GeoSpace>);
  using coordinate_type             = typename GeoSpace::coordinate_type;
  static constexpr size_t dimension = GeoSpace::dimension;

 public:
  constexpr pointnd() = default;
  constexpr pointnd(const coordinate_type& v0, const coordinate_type& v1)
      : _values{v0, v1} {}
  constexpr pointnd(std::initializer_list<coordinate_type> init) {
    std::copy(init.begin(), init.end(), _values);
  }

  constexpr coordinate_type& operator[](size_t pos) { return _values[pos]; }
  constexpr const coordinate_type& operator[](size_t pos) const {
    return _values[pos];
  }

  constexpr coordinate_type& at(size_t pos) {
    if (pos >= dimension) {
      throw std::out_of_range("Point coordinate visit out of dimension range");
    }
    return _values[pos];
  }

  constexpr const coordinate_type& at(size_t pos) const {
    if (pos >= dimension) {
      throw std::out_of_range("Point coordinate visit out of dimension range");
    }
    return _values[pos];
  }

  coordinate_type&                 x() { return at(0); }
  coordinate_type&                 y() { return at(1); }
  coordinate_type&                 z() { return at(2); }
  coordinate_type&                 m() { return at(3); }
  constexpr const coordinate_type& x() const { return at(0); }
  constexpr const coordinate_type& y() const { return at(1); }
  constexpr const coordinate_type& z() const { return at(2); }
  constexpr const coordinate_type& m() const { return at(3); }

  void x(const coordinate_type& v) { at(0) = v; }
  void y(const coordinate_type& v) { at(1) = v; }
  void z(const coordinate_type& v) { at(2) = v; }
  void m(const coordinate_type& v) { at(3) = v; }

  bool operator<(const pointnd& other) const {
    for (std::size_t i = 0; i < dimension; ++i) {
      if (_values[i] >= other._values[i])
        return false;
    }
    return true;
  }
  bool operator>(const pointnd& other) const { return other < *this; }
  bool operator<=(const pointnd& other) const {
    for (std::size_t i = 0; i < dimension; ++i) {
      if (_values[i] > other._values[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator>=(const pointnd& other) const { return other <= *this; }
  bool operator==(const pointnd& other) const {
    return std::equal(_values, _values + dimension, other._values,
                      other._values + dimension);
  }
  bool operator!=(const pointnd& other) const { return !(*this == other); }

 private:
  coordinate_type _values[dimension];
};

}  // namespace odrc::geometry