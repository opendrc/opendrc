#pragma once

#include <odrc/core/database.hpp>

namespace odrc {
void area_check_seq(const odrc::core::database& db, int layer, int threshold);
void area_check_dac23(const odrc::core::database& db, int layer, int threshold);
}  // namespace odrc