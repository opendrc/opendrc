#include <iostream>

#include <odrc/interface/gdsii/gdsii.hpp>

void help() {
  std::cerr << "Usage: ./odrc <gds_in>" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    help();
    return 2;
  }
  odrc::gdsii_library lib;
  try {
    lib.read(argv[1]);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}