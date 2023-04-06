#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <odrc/db/cell.hpp>
#include <odrc/db/layer.hpp>
#include <odrc/utility/datetime.hpp>
#include <odrc/utility/exception.hpp>
#include <odrc/utility/proxy.hpp>

namespace odrc::db {

template <typename Cell = cell<>, typename Layer = layer<>>
class database {
 public:
  database() = default;

  // attributes

  [[nodiscard]] int get_version() const noexcept { return _version; }
  [[nodiscard]] const std::string& get_name() const noexcept { return _name; }
  [[nodiscard]] const odrc::util::datetime& get_mtime() const noexcept {
    return _mtime;
  }
  [[nodiscard]] const odrc::util::datetime& get_atime() const noexcept {
    return _atime;
  }
  [[nodiscard]] double get_dbu_in_user_unit() const noexcept {
    return _dbu_in_user_unit;
  }
  [[nodiscard]] double get_dbu_in_meter() const noexcept {
    return _dbu_in_meter;
  }

  void set_version(int version) { _version = version; }
  void set_name(const std::string& name) { _name = name; }
  void set_mtime(const odrc::util::datetime& mtime) { _mtime = mtime; }
  void set_atime(const odrc::util::datetime& atime) { _atime = atime; }
  void set_dbu_in_user_unit(double dbu_in_user_unit) {
    _dbu_in_user_unit = dbu_in_user_unit;
  }
  void set_dbu_in_meter(double dbu_in_meter) { _dbu_in_meter = dbu_in_meter; }

  // member access

  Cell& get_cell(int id) { return _cells.at(id); }
  Cell& get_cell(const std::string& name) {
    int id = get_cell_id(name);
    return _cells.at(id);
  }

  [[nodiscard]] const Cell& get_cell(int id) const { return _cells.at(id); }
  [[nodiscard]] const Cell& get_cell(const std::string& name) const {
    int id = get_cell_id(name);
    return _cells.at(id);
  }
  [[nodiscard]] int get_cell_id(const std::string& name) const {
    return _name_to_id.at(name);
  }

  Cell& create_cell(const std::string_view name = "") {
    int   id   = _cells.size();
    auto& cell = _cells.emplace_back(id, name);
    return cell;
  }

  // iterators

  [[nodiscard]] auto cells() const {
    return odrc::util::const_container_proxy(_cells);
  }
  [[nodiscard]] auto layers() const {
    return odrc::util::const_container_proxy(_layers);
  }

  // operations

  void update_map() {
    // update map if it's not up-to-date
    // do nothing if both containers have equal sizes
    for (auto i = _name_to_id.size(); i < _cells.size(); ++i) {
      _name_to_id.emplace(_cells.at(i).get_name(), i);
    }
  }

  void construct_layers(int top_cell_id = -1) {
    Cell& top_cell = top_cell_id == -1 ? _cells.back() : _cells.at(top_cell_id);

    for (auto& elem : top_cell.elements()) {
      int  l    = elem.get_layer();
      auto iter = _layers.find(l);
      if (iter == _layers.end()) {
        _layers.emplace(l, Layer(l));
      }
      _layers.at(l).add(elem);
    }

    for (auto& cell_ref : top_cell.cell_refs()) {
      auto& cell      = get_cell(cell_ref.get_id());
      auto& ref_point = cell_ref.get_ref_point();
      if (cell.num_cell_refs() != 0) {
        throw odrc::invalid_file(
            "Hierarchy with more than two layers is not supported");
      }
      for (auto& elem : cell.elements()) {
        int  l    = elem.get_layer();
        auto iter = _layers.find(l);
        if (iter == _layers.end()) {
          _layers.emplace(l, Layer(l));
        }
        _layers.at(l).add(elem + ref_point);
      }
    }
  }
  void construct_layers(const std::string& top_cell_name) {
    this->contruct_layers(_name_to_id.at(top_cell_name));
  }

 private:
  // meta info
  int                  _version     = -1;
  int                  _top_cell_id = -1;
  std::string          _name;
  odrc::util::datetime _mtime;
  odrc::util::datetime _atime;
  double               _dbu_in_user_unit;
  double               _dbu_in_meter;

  std::vector<Cell>              _cells;
  std::unordered_map<int, Layer> _layers;

  std::unordered_map<std::string, int> _name_to_id;
};
}  // namespace odrc::db