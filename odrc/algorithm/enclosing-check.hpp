#pragma once

#include <odrc/core/database.hpp>
namespace odrc {
void enclosing_check_seq(const odrc::core::database& db,
                         std::vector<int>            layers,
                         int                         threshold,
                         rule_type                   ruletype,
                         std::vector<check_result>&  vios);
}