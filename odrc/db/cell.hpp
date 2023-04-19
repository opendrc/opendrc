#pragma once

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
template <typename Element = element<>>
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
  void clear() {
    _ref_cell_id            = -1;
    _ref_point              = point_t{};
    _transform.is_reflected = false;
    _transform.is_magnified = false;
    _transform.is_rotated   = false;
    _transform.mag          = 1.0;
    _transform.angle        = 0.0;
  }

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

template <typename Element>
class cell {
 public:
  using element_t    = Element;
  using cell_ref_t   = cell_ref<cell>;
  using elements_t   = std::vector<element_t>;
  using cell_refs_t  = std::vector<cell_ref_t>;
  using layer_view_t = std::unordered_map<int, elements_t>;

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
      int         layer,
      std::size_t index) const noexcept {
    return _layer_view.at(layer).first.at(index);
  }
  [[nodiscard]] constexpr const cell_ref_t& get_cell_ref(
      int         layer,
      std::size_t index) const noexcept {
    return _layer_view.at(layer).second.at(index);
  }

  // iterators

  [[nodiscard]] auto elements() const noexcept {
    return odrc::util::const_container_proxy(_layer_view);
  }
  [[nodiscard]] auto elements_on_layer(int layer) const {
    return odrc::util::const_container_proxy(_layer_view.at(layer).first);
  }
  [[nodiscard]] auto cell_refs() const noexcept {
    return odrc::util::const_container_proxy(_cell_refs);
  }

  // states

  [[nodiscard]] constexpr std::size_t num_layers() const noexcept {
    return _layer_view.size();
  }

  [[nodiscard]] constexpr bool has_layer(int layer) const noexcept {
    return _layer_view.find(layer) != _layer_view.end();
  }

  // Note: the cost of the function is O(#layer)
  [[nodiscard]] constexpr std::size_t num_elements() const noexcept {
    std::size_t num = 0;
    for (const auto& [layer, objects] : _layer_view) {
      num += objects.size();
    }
    return num;
  }

  [[nodiscard]] constexpr std::size_t num_cell_refs() const noexcept {
    return _cell_refs.size();
  }

  // Note: the cost of the function is O(#layer)
  [[nodiscard]] constexpr std::size_t num_objects() const noexcept {
    return num_elements() + num_cell_refs();
  }

  [[nodiscard]] constexpr std::size_t num_elements_on_layer(
      int layer) const noexcept {
    auto iter = _layer_view.find(layer);
    if (iter == _layer_view.end()) {
      return 0;
    }
    return iter->second.size();
  }

  // modifiers

  void clear() {
    _layer_view.clear();
    _id   = -1;
    _name = "";
  }

  void insert(const Element& elem) {
    int  layer = elem.get_layer();
    auto iter  = _layer_view.find(layer);
    if (iter == _layer_view.end()) {
      iter = _layer_view.emplace(layer, elements_t{}).first;
    }
    iter->second.emplace_back(elem);
  }

  void insert(Element&& elem) {
    int  layer = elem.get_layer();
    auto iter  = _layer_view.find(layer);
    if (iter == _layer_view.end()) {
      iter = _layer_view.emplace(layer, elements_t{}).first;
    }
    iter->second.emplace_back(std::move(elem));
  }

  void insert(const cell_ref_t& cref) { _cell_refs.emplace_back(cref); }

  void insert(cell_ref_t&& cref) { _cell_refs.emplace_back(std::move(cref)); }

 private:
  // meta info
  int                  _id;
  std::string          _name;
  odrc::util::datetime _mtime;
  odrc::util::datetime _atime;

  // data
  layer_view_t _layer_view;
  cell_refs_t  _cell_refs;
};

}  // namespace odrc::db