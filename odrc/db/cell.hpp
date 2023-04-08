#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <odrc/db/elem.hpp>
#include <odrc/utility/datetime.hpp>
#include <odrc/utility/proxy.hpp>

namespace odrc::db {

// forward declaration
template <typename Element = element<>,
          template <typename T, typename Allocator> typename Container =
              std::vector,
          template <typename T> typename Allocator = std::allocator>
class cell;

template <typename Cell = cell<>>
class cell_ref {
 public:
  using point_t = typename Cell::element_t::polygon_t::vertex;
  struct transform {
    bool   is_reflected;
    bool   is_magnified;
    bool   is_rotated;
    double mag;
    double angle;
  };

  cell_ref() = default;
  cell_ref(int cell_id, const point_t& ref_point)
      : _ref_cell_id(cell_id), _ref_point(ref_point) {}

  [[nodiscard]] int            get_id() const noexcept { return _ref_cell_id; }
  [[nodiscard]] const point_t& get_ref_point() const noexcept {
    return _ref_point;
  }
  void set_id(int id) { _ref_cell_id = id; }
  void set_ref_point(const point_t& ref_point) { _ref_point = ref_point; }

  // state
  [[nodiscard]] bool is_reflected() const noexcept {
    return _transform.is_reflected;
  }
  [[nodiscard]] bool is_magnified() const noexcept {
    return _transform.is_magnified;
  }
  [[nodiscard]] bool is_rotated() const noexcept {
    return _transform.is_rotated;
  }
  [[nodiscard]] double get_mag() const noexcept { return _transform.mag; }
  [[nodiscard]] double get_angle() const noexcept { return _transform.angle; }
  [[nodiscard]] const transform& get_transform() const noexcept {
    return _transform;
  }

  // operations
  void reflect() { _transform.is_reflected = true; }
  void magnify(double mag) {
    _transform.is_magnified = true;
    _transform.mag          = mag;
  }
  void rotate(double angle) {
    _transform.is_rotated = true;
    _transform.angle      = angle;
  }

 private:
  int       _ref_cell_id = -1;
  point_t   _ref_point   = point_t{};
  transform _transform   = transform{false, false, false, 1.0, 0.0};
};

namespace traits {

template <typename T, typename V, typename = void>
struct has_emplace_back : std::false_type {};

template <typename T, typename V>
struct has_emplace_back<
    T,
    V,
    std::void_t<decltype(std::declval<T>().emplace_back(std::declval<V>()))>>
    : std::true_type {};

template <typename T, typename V>
inline constexpr bool has_emplace_back_v = has_emplace_back<T, V>::value;

}  // namespace traits

template <typename Element,
          template <typename T, typename Allocator>
          typename Container,
          template <typename T>
          typename Allocator>
class cell {
 public:
  using element_t     = Element;
  using element_list  = Container<Element, Allocator<Element>>;
  using cell_ref_t    = cell_ref<cell>;
  using cell_ref_list = Container<cell_ref_t, Allocator<cell_ref_t>>;
  using layer_index_t = std::unordered_map<int, std::vector<std::size_t>>;

  constexpr cell() = default;
  constexpr cell(int id) noexcept : _id(id) {}
  constexpr cell(const std::string_view name) : _name(name) {}
  constexpr cell(int id, const std::string_view name) : _id(id), _name(name) {}

  // attributes

  [[nodiscard]] constexpr int get_id() const noexcept { return _id; }
  [[nodiscard]] constexpr const std::string_view get_name() const noexcept {
    return _name;
  }
  [[nodiscard]] constexpr const odrc::util::datetime& get_mtime()
      const noexcept {
    return _mtime;
  }
  [[nodiscard]] constexpr const odrc::util::datetime& get_atime()
      const noexcept {
    return _atime;
  }
  void set_id(int id) noexcept { _id = id; }
  void set_name(const std::string_view name) { _name = name; }
  void set_mtime(const odrc::util::datetime& mtime) noexcept { _mtime = mtime; }
  void set_atime(const odrc::util::datetime& atime) noexcept { _atime = atime; }

  // member access

  [[nodiscard]] constexpr const element_t& get_element(
      std::size_t index) const noexcept {
    return _elements.at(index);
  }
  [[nodiscard]] constexpr const cell_ref_t& get_cell_ref(
      std::size_t index) const noexcept {
    return _cell_refs.at(index);
  }

  element_t& create_element() {
    if constexpr (traits::has_emplace_back_v<element_list, element_t>) {
      _elements.emplace_back();
    } else {
      _elements.insert(_elements.end(), element_t{});
    }
    return _elements.back();
  }

  cell_ref_t& create_cell_ref() {
    if constexpr (traits::has_emplace_back_v<cell_ref_list, cell_ref_t>) {
      _cell_refs.emplace_back();
    } else {
      _cell_refs.insert(_cell_refs.end(), cell_ref_t{});
    }
    return _cell_refs.back();
  }

  // iterators

  [[nodiscard]] auto elements() const {
    return odrc::util::const_container_proxy(_elements);
  }
  [[nodiscard]] auto cell_refs() const {
    return odrc::util::const_container_proxy(_cell_refs);
  }

  // states

  [[nodiscard]] constexpr std::size_t num_elements() const noexcept {
    return _elements.size();
  }
  [[nodiscard]] constexpr std::size_t num_cell_refs() const noexcept {
    return _cell_refs.size();
  }

  // modifiers

  void add(const Element& elem) {
    int  layer = elem.get_layer();
    auto iter  = _layer_index.find(layer);
    if (iter == _layer_index.end()) {
      _layer_index.emplace(layer, std::vector{_elements.size()});
    } else {
      iter->second.emplace_back(layer);
    }

    if constexpr (traits::has_emplace_back_v<element_list, element_t>) {
      _elements.emplace_back(elem);
    } else {
      _elements.insert(_elements.end(), elem);
    }
  }

 private:
  // meta info
  int                  _id;
  std::string          _name;
  odrc::util::datetime _mtime;
  odrc::util::datetime _atime;

  // data
  element_list  _elements;
  cell_ref_list _cell_refs;

  // reverse index
  layer_index_t _layer_index;
};

// TODO: a cell_view returns objects within a specific layer
// c++20 filter_view
template <typename Cell>
class cell_layer_view {
 public:
  cell_layer_view(const Cell& cell, int layer)
      : _cell(cell), _layer(layer), _cached(false) {}

 private:
  const Cell& _cell;
  int         _layer  = -1;
  bool        _cached = false;
};
}  // namespace odrc::db