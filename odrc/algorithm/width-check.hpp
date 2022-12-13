#pragma once

#include <odrc/core/common_structs.hpp>
#include <odrc/core/database.hpp>
namespace odrc {
void width_check_par(const odrc::core::database& db,
                     int                         layer,
                     int                         threshold,
                     std::vector<check_result>&  vios);
void width_check_seq(const odrc::core::database& db,
                     int                         layer,
                     int                         threshold,
                     std::vector<check_result>&  vios);
}  // namespace odrc