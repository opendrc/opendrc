#pragma once

#include <odrc/core/common_structs.hpp>
#include <odrc/core/database.hpp>
namespace odrc {
void space_check_seq(odrc::core::database&      db,
                     std::vector<int>           layers,
                     std::vector<int>           without_layer,
                     int                        threshold,
                     rule_type                  ruletype,
                     std::vector<check_result>& vios);

void space_check_par(const odrc::core::database& db,
                     int                         layer1,
                     int                         layer2,
                     int                         threshold,
                     std::vector<check_result>&  vios);
}  // namespace odrc