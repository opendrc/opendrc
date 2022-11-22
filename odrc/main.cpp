#include <iostream>

#include <odrc/algorithm/area-check.hpp>
#include <odrc/algorithm/space-check.hpp>
#include <odrc/algorithm/width-check.hpp>
#include <odrc/gdsii/gdsii.hpp>

#include <odrc/utility/timer.hpp>

void help() {
  std::cerr << "Usage: ./odrc <gds_in>" << std::endl;
}

int main(int argc, char* argv[]) {
  int layer1 = 20; // 19, 20, 30
  int layer2 = 25; // 21, 25, 35
  if (argc < 2) {
    help();
    return 2;
  } else if(argc == 4) {
    layer1 = std::stoi(argv[2]);
    layer2 = std::stoi(argv[3]);

  }
  odrc::core::database db;
  try {
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t("t", logger);
      t.start();
      db = odrc::gdsii::read(argv[1]);
      t.pause();
    }

    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t("t", logger);
      t.start();
      db.update_depth_and_mbr(layer1, layer2);
      t.pause();
    }
    // for(auto &c: db.cells) {
    //   std::cout << c.name << ": " << c.depth << std::endl;
    // }
    // return 0;
    // {
    //   odrc::util::logger logger("/dev/null", odrc::util::log_level::info,
    //   true); odrc::util::timer  t("t", logger); t.start();
    //   odrc::width_xcheck(db, 19, 18);
    //   t.pause();
    // }
    // {
    //   odrc::util::logger logger("/dev/null", odrc::util::log_level::info,
    //   true); odrc::util::timer  t("t", logger); t.start();
    //   odrc::width_xcheck(db, 20, 18);
    //   t.pause();
    // }
    // {
    //   odrc::util::logger logger("/dev/null", odrc::util::log_level::info,
    //   true); odrc::util::timer  t("t", logger); t.start();
    //   odrc::width_xcheck(db, 30, 18);
    //   t.pause();
    // }
    // odrc::width_check(db, 11, 650);
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t1("area1", logger);
      odrc::util::timer  t2("area2", logger);
      odrc::util::timer  t3("area3", logger);
      t1.start();
      odrc::area_check_dac23(db, 19, 504);
      t1.pause();
      t2.start();
      odrc::area_check_dac23(db, 20, 504);
      t2.pause();
      t3.start();
      odrc::area_check_dac23(db, 30, 504);
      t3.pause();
    }
    // {
    //   odrc::util::logger logger("/dev/null", odrc::util::log_level::info,
    //   true); odrc::util::timer  t("t", logger); t.start();
    //   odrc::area_check_cpu(db, 19, 10000);
    //   t.pause();
    // }

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}