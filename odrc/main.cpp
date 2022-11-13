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
      db.update_depth_and_mbr();
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
    // {
    //   odrc::util::logger logger("/dev/null", odrc::util::log_level::info,
    //   true); odrc::util::timer  t("t", logger); t.start();
    //   odrc::space_check_dac23(db, 19, 19, 18);
    //   t.pause();
    // }
#ifdef AREA_TEST
    {
      odrc::util::logger logger("/dev/null", odrc::util::log_level::info, true);
      odrc::util::timer  t("t", logger);
      t.start();
      odrc::area_check_cpu(db, 19, 10000);
      t.pause();
    }
#endif

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}