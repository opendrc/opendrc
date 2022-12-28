#include <iostream>

#include <functional>
#include <odrc/core/engine.hpp>
#include <odrc/gdsii/gdsii.hpp>
using mode = odrc::core::mode;
using cell = odrc::core::cell;
void help() {
  std::cerr << "Usage: ./odrc <gds_in>" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    help();
    return 2;
  }
  try {
    auto e = odrc::core::engine();
    e.add_rules({
        // e.polygons().is_rectilinear(),
        // e.layer(20).width().greater_than(18),
         e.layer(19).spacing().greater_than(18000000),
        //e.layer(19).with_layer(21).enclosure().greater_than(5),
        // e.layer(19).area().greater_than(504)
        //  e.layer(19).polygons().ensures(
        //      [](const auto& p) { return !p.name.empty(); })
    });
    e.set_mode(mode::sequential);
    auto db = odrc::gdsii::read(argv[1]);
    e.check(db);
    // e.set_mode(mode::parallel);
    // e.check(db);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}