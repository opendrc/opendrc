#pragma once

#include <odrc/core/database.hpp>

namespace odrc {
void space_check(const odrc::core::database& db,
                 int                         layer1,
                 int                         layer2,
                 int                         threshold);
void space_check_dac23(const odrc::core::database& db,
                       int                         layer1,
                       int                         layer2,
                       int                         threshold);
void enclosing_check_dac23(const odrc::core::database& db,
                           int                         layer1,
                           int                         layer2,
                           int                         threshold);
}  // namespace odrc