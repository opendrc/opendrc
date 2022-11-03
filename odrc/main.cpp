#include <iostream>
#include <odrc/algorithm/space-check.hpp>
#include <odrc/algorithm/width-check.hpp>
#include <odrc/gdsii/gdsii.hpp>
#include <odrc/core/overlap_check.hpp>
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
    db.update_depth_and_mbr();
    odrc::core::overlap_check(db,7);
    odrc::width_check(db, 11, 650);
    odrc::space_check(db, 11, 11, 650);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}