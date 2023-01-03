#pragma once

#include <odrc/algorithm/sequential_mode.hpp>
#include <odrc/core/database.hpp>
namespace odrc {

void space_check_par(const odrc::core::database& db,
                     int                         layer1,
                     int                         layer2,
                     int                         threshold,
                     std::vector<violation_information>&  vios);
void width_check_seq(const odrc::core::database& db,
                     int                         layer,
                     int                         threshold,
                     std::vector<violation_information>&  vios);

}  // namespace odrc