#include <iostream>

#include <odrc/core/engine.hpp>
#include <odrc/gdsii/gdsii.hpp>

using mode = odrc::core::mode;

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
        /* examples for add rules
           e.polygons().is_rectilinear(),
           e.layer(20).width().greater_than(18),
           e.layer(30).width().greater_than(18),
           e.layer(20).spacing().greater_than(18),
        */
        e.layer(19).width().greater_than(18),
        // e.layer(19).spacing().greater_than(18),
        // e.layer(19).with_layer(21).enclosure().greater_than(2),
        // e.layer(19).area().greater_than(504),
    });
    e.set_mode(mode::sequential);
    auto db = odrc::gdsii::read(argv[1]);
    e.check(db);
    // for parallel mode
    e.set_mode(mode::parallel);
    e.check(db);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}