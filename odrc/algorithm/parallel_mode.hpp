#pragma once

#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <odrc/core/edge.hpp>
#include <odrc/core/interval_tree.hpp>
#include <odrc/core/rule.hpp>

namespace odrc {
struct coord {
  int x;
  int y;
};
struct check_result {
  int  e11x;
  int  e11y;
  int  e12x;
  int  e12y;
  int  e21x;
  int  e21y;
  int  e22x;
  int  e22y;
  bool is_violation = false;
};

// transform intra-cell violations to inter-cell violations
inline void transform_vio(const check_result&           vio,
                          const core::coord&            offset,
                          std::vector<core::violation>& vios) {
  if (vio.is_violation) {
    core::edge edge1{{vio.e11x + offset.x, vio.e11y + offset.y},
                     {vio.e12x + offset.x, vio.e12y + offset.y}};
    core::edge edge2{{vio.e21x + offset.x, vio.e21y + offset.y},
                     {vio.e22x + offset.x, vio.e22y + offset.y}};
    vios.emplace_back(edge1, edge2);
  }
}

inline void get_ref_vios(
    odrc::core::database&                                 db,
    std::unordered_map<std::string, std::pair<int, int>>& check_results,
    check_result*                                         check_results_host,
    const odrc::core::cell&                               cell,
    std::vector<core::violation>&                         vios) {
  for (auto i = 0UL; i < cell.cell_refs.size(); i++) {
    auto name = cell.cell_refs.at(i).cell_name;
    if (check_results.find(name) != check_results.end()) {
      auto [begin, end] = check_results.at(name);
      for (int i = begin; i < end; i++) {
        transform_vio(check_results_host[i], cell.cell_refs.at(i).ref_point,
                      vios);
      }
    }
  }
}

void space_check_par(const odrc::core::database&   db,
                     core::rule                    rule,
                     std::vector<core::violation>& vios);
void area_check_par(const odrc::core::database&   db,
                    core::rule                    rule,
                    std::vector<core::violation>& vios);
void width_check_par(odrc::core::database&         db,
                     int                           layer,
                     int                           threshold,
                     std::vector<core::violation>& vios);
void enc_check_par(const odrc::core::database&   db,
                   core::rule                    rule,
                   std::vector<core::violation>& vios);

}  // namespace odrc