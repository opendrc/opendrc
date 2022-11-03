#include <iostream>

#include <odrc/algorithm/space-check.hpp>
#include <odrc/algorithm/width-check.hpp>
#include <odrc/gdsii/gdsii.hpp>

#include <odrc/utility/timer.hpp>

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
    // db.update_depth_and_mbr();
    // for(auto &c: db.cells) {
    //   std::cout << c.name << ": " << c.depth << std::endl;
    // }
    // return 0;
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t("t", logger);
      t.start();
      odrc::width_check_cpu(db, 20, 18);
      t.pause();
    }
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t("t", logger);
      t.start();
      odrc::width_check_cpu(db, 30, 18);
      t.pause();
    }
    // odrc::width_check(db, 11, 650);
    // odrc::space_check(db, 11, 11, 650);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}