#pragma once

#include <odrc/core/common_structs.hpp>
#include <odrc/core/database.hpp>

namespace odrc {
void enclosing_check_seq(odrc::core::database&      db,
                         std::vector<int>           layers,
                         std::vector<int>           without_layer,
                         int                        threshold,
                         rule_type                  ruletype,
                         std::vector<check_result>& vios);
}