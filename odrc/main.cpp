#include <iostream>

#include <odrc/algorithm/width-check.hpp>
#include <odrc/interface/gdsii/gdsii.hpp>

void help() {
  std::cerr << "Usage: ./odrc <gds_in>" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    help();
    return 2;
  }
  odrc::gdsii::library lib;
  try {
    lib.read(argv[1]);
    odrc::width_check(lib, 11, 650);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}