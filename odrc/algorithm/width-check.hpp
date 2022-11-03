#pragma once

#include <odrc/core/database.hpp>

namespace odrc {
void width_check(const odrc::core::database& db, int layer, int threshold);
void width_check_cpu(const odrc::core::database& db, int layer, int threshold);
}