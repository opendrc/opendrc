#pragma once

#include <vector>

#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <odrc/core/edge.hpp>
#include <odrc/core/interval_tree.hpp>
#include <odrc/core/rule.hpp>

namespace odrc {
using coord = odrc::core::coord;
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
struct evnt {
  int x;
  int y1;
  int y2;
  int id;
};

inline bool is_violation(core::orthogonal_edge& edge1,
                         core::orthogonal_edge& edge2,
                         int                    threshold) {
  auto [p1_start, p1_end, intercept1] = edge1;
  auto [p2_start, p2_end, intercept2] = edge2;
  if (std::abs(intercept1 - intercept2) < threshold) {
    return not(p1_start >= p2_end or p2_start >= p1_end);
  } else {
    return false;
  }
}
// transform intra-cell violations to inter-cell violations
inline void transform_vio(const check_result&           vio,
                          const coord&                  offset,
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

inline void result_transform(std::vector<core::violation>& vios,
                             check_result*                 results,
                             int                           num) {
  for (int i = 0; i < num; i++) {
    if (results[i].is_violation) {
      core::edge edge1{{results[i].e11x, results[i].e11y},
                       {results[i].e12x, results[i].e12y}};
      core::edge edge2{{results[i].e21x, results[i].e21y},
                       {results[i].e22x, results[i].e22y}};
      vios.emplace_back(edge1, edge2);
    }
  }
}

void space_check_par(odrc::core::database&         db,
                     int                           layer1,
                     int                           threshold,
                     std::vector<core::violation>& vios);
void width_check_par(odrc::core::database&         db,
                     int                           layer,
                     int                           threshold,
                     std::vector<core::violation>& vios);
void enc_check_par(odrc::core::database&         db,
                   int                           layer1,
                   int                           layer2,
                   int                           threshold,
                   std::vector<core::violation>& vios);
void area_check_par(odrc::core::database&         db,
                    int                           layer,
                    int                           threshold,
                    std::vector<core::violation>& vios);
}  // namespace odrc