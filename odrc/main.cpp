#include <iostream>

#include <odrc/algorithm/space-check.hpp>
#include <odrc/algorithm/width-check.hpp>
#include <odrc/core/engine.hpp>
#include <odrc/gdsii/gdsii.hpp>
void help() {
  std::cerr << "Usage: ./odrc <gds_in>" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    help();
    return 2;
  }
  try {
    auto db = odrc::gdsii::read(argv[1]);
    auto e  = odrc::core::engine();
    e.add_rules({
        e.polygons().is_rectilinear(), e.layer(19).width().greater_than(18)
        // e.layer(20).width().ensures(
        //     [](const auto& p) { return !p.name.empty();
    });
    e.check(db);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}