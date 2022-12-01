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
    auto               db = odrc::gdsii::read(argv[1]);
    odrc::core::engine e(db);
    e.rectilinear();
    // e.width(1,std::pair<108,NULL>);
    // e.width(1, std::pair<54, NULL>);
    // e.spacing(1,std::pair<108,NULL>);
    // e.spacing(1, std::pair<54, NULL>);
    // e.area(1, 5832);
    // e.extension(1, 7, std::pair<7, NULL>,10);

    // e.width(2, std::pair<7, 7>);
    // e.width(2,std::pair<108,NULL>);
    // e.not_bend(2);

    // e.spacing(7,std::pair<27, NULL>);
    // e.not_bend(7);
    // e.not_overlapping(7, 11);
    // e.spacing(10,11,std::pair<4,NULL>)
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}