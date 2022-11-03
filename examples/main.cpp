#include <iostream>

#include <odrc/gdsii/gdsii.hpp>

#include "overlap_query.cpp"

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
    examples::overlap_query(db, 7);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
