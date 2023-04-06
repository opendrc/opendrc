#pragma once

namespace odrc::util {
template <typename Container>
class const_container_proxy {
 public:
  const_container_proxy(const Container& container) : _container(container) {}
  [[nodiscard]] typename Container::const_iterator begin() const noexcept {
    return _container.begin();
  };
  [[nodiscard]] typename Container::const_iterator end() const noexcept {
    return _container.end();
  };

  [[nodiscard]] const auto& at(typename Container::size_type n) const {
    return _container.at(n);
  }
  [[nodiscard]] const auto& operator[](
      typename Container::size_type n) const noexcept {
    return _container[n];
  }

  [[nodiscard]] typename Container::size_type size() const noexcept {
    return _container.size();
  }

 private:
  const Container& _container;
};
}  // namespace odrc::util