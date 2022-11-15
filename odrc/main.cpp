// #define AREA_TEST
#include <iostream>

#include <odrc/algorithm/space-check.hpp>
#include <odrc/algorithm/width-check.hpp>
#include <odrc/algorithm/area-check.hpp>
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
  int                  layer1 = 20;
  int                  layer2 = 20;
  if(argc == 3) {
    layer1 = std::stoi(argv[2]);
    layer2 = std::stoi(argv[2]);
  } else if(argc == 4) {
    layer1 = std::stoi(argv[2]);
    layer2 = std::stoi(argv[3]);
  }

  odrc::core::database db;
  try {
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t0("t0", logger);
      t0.start();
      db = odrc::gdsii::read(argv[1]);
      t0.pause();
    }

    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t("t", logger);
      t.start();
      db.update_depth_and_mbr(layer1, layer2);
      t.pause();
    }
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info,true);
      odrc::util::timer  space("space", logger); 
      space.start();
      odrc::space_check_dac23(db, layer1, layer2,5);
      //odrc::enclosing_check_dac23(db, layer1, layer2,5);
      space.pause();
    }
#ifdef AREA_TEST
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t("t", logger);
      t.start();
      odrc::area_check_cpu(db, 20, 10000);
      t.pause();
    }
#endif

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}