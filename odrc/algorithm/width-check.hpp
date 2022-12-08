#pragma once

#include <odrc/core/database.hpp>

namespace odrc {
void width_check_pal(const odrc::core::database& db,
                     int                         layer,
                     int                         threshold,
                     std::vector<check_result>&  vios);
void width_check_seq(const odrc::core::database& db,
                     int                         layer,
                     int                         threshold,
                     std::vector<check_result>&  vios);
}  // namespace odrc