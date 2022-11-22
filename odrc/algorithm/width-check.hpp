#pragma once

#include <odrc/core/database.hpp>

namespace odrc {
void width_check(const odrc::core::database& db, int layer, int threshold);
void width_check_cpu(const odrc::core::database& db, int layer, int threshold);
void width_check_dac23(const odrc::core::database& db, int layer, int threshold);
void width_xcheck(const odrc::core::database& db, int layer, int threshold);
}