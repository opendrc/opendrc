#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <odrc/db/cell.hpp>
#include <odrc/db/layer.hpp>
#include <odrc/utility/exception.hpp>
#include <odrc/utility/proxy.hpp>

namespace odrc::db {

template <typename Cell = cell<>, typename Layer = layer<>>
class database {
 public:
  Cell& create_cell(const std::string_view name) {
    int   id   = _cells.size();
    auto& cell = _cells.emplace_back(id, name);
    _name_to_id.emplace(name, id);
    return cell;
  }
  Cell& get_cell(int id) { return _cells.at(id); }
  Cell& get_cell(const std::string& name) {
    int id = _name_to_id.at(name);
    return _cells.at(id);
  }

  [[nodiscard]] const Cell& get_cell(int id) const { return _cells.at(id); }
  [[nodiscard]] const Cell& get_cell(const std::string& name) const {
    int id = _name_to_id.at(name);
    return _cells.at(id);
  }

  [[nodiscard]] auto cells() const {
    return odrc::util::const_container_proxy(_cells);
  }
  [[nodiscard]] auto layers() const {
    return odrc::util::const_container_proxy(_layers);
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
  std::vector<Cell>              _cells;
  std::unordered_map<int, Layer> _layers;

  std::unordered_map<std::string, int> _name_to_id;
};
}  // namespace odrc::db