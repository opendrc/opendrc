#pragma once

#include <odrc/core/database.hpp>
namespace odrc {
struct check_result {
  int  e11x;
  int  e11y;
  int  e12x;
  int  e12y;
  int  e21x;
  int  e21y;
  int  e22x;
  int  e22y;
  bool is_violation = false;
};
void space_check_pal(const odrc::core::database& db,
                     int                         layer1,
                     int                         layer2,
                     int                         threshold,
                     std::vector<check_result>&  vios);

void space_check_seq(const odrc::core::database& db,
                     int                         layer1,
                     int                         layer2,
                     int                         threshold,
                     std::vector<check_result>&  vios);
}  // namespace odrc