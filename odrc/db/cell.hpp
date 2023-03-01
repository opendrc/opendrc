#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <odrc/db/elem.hpp>

namespace odrc::db {

// forward declaration
template <typename Element = element<>,
          template <typename T, typename Allocator> typename Container =
              std::vector,
          template <typename T> typename Allocator = std::allocator>
class cell;

template <typename Cell>
class cell_ref {
 public:
  using point           = typename Cell::element_t::polygon_type::point_type;
  using cell_shared_ptr = std::shared_ptr<Cell>;
  using cell_weak_ptr   = std::weak_ptr<Cell>;

  constexpr cell_ref(cell_shared_ptr cell_shared, const point& p = point{})
      : _cell_weak(cell_shared), _ref_point(p) {}

  Cell* get() const noexcept { return _cell_weak.lock().get(); }

  Cell& operator*() const noexcept { return *_cell_weak.lock(); }
  Cell* operator->() const noexcept { return _cell_weak.lock().operator->(); }

 private:
  cell_weak_ptr _cell_weak;
  point         _ref_point;
};

template <typename T, typename V, typename = void>
struct has_emplace_back : std::false_type {};

template <typename T, typename V>
struct has_emplace_back<
    T,
    V,
    std::void_t<decltype(std::declval<T>().emplace_back(std::declval<V>()))>>
    : std::true_type {};

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

  constexpr cell() = default;
  void add(const Element& elem) {
    if constexpr (has_emplace_back<element_list, element_t>::value) {
      _elements.emplace_back(elem);
    } else {
      _elements.insert(_elements.end(), elem);
    }
  }
  constexpr std::size_t num_elements() const noexcept {
    return _elements.size();
  }

 private:
  element_list  _elements;
  cell_ref_list _cell_refs;
};
}  // namespace odrc::db