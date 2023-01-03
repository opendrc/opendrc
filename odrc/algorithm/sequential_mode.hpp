#pragma once

#include <limits>
#include <odrc/core/database.hpp>
#include <odrc/core/interval_tree.hpp>
#include "odrc/core/cell.hpp"
namespace odrc {
using Intvl = core::interval<int, int>;

struct event {
  Intvl intvl;
  int   y;
  bool  is_polygon;
  bool  is_inevent;
};

struct violation_information {
  const core::polygon* polygon1;
  const core::polygon* polygon2;
  core::edge           edge1;
  core::edge           edge2;
  violation_information(const core::polygon* polygon) {
    this->polygon1 = polygon;
  };
  violation_information(const core::polygon* polygon1,
                        const core::polygon* polygon2) {
    this->polygon1 = polygon1;
    this->polygon2 = polygon2;
  };
  violation_information(core::edge edge1, core::edge edge2) {
    this->edge1 = edge1;
    this->edge2 = edge2;
  };
};

enum class rule_type {
  enclosure          = 0,
  spacing_both       = 1,  // default,both horizontal and vertical edges
  spacing_h_edge     = 2,  // only horizontal edges
  spacing_v_edge     = 3,  // only vertical edges
  spacing_corner     = 4,  // corner-corner
  spacing_center     = 5,  // center-center
  spacing_tip        = 6,  // tip-tip
  spacing_lup        = 7,  // latch-upobject
  extension          = 8,
  geometry           = 9,
  length             = 10,
  lup                = 11,
  overlap            = 12,
  recommend          = 13,  // may be unused
  area               = 14,
  sram               = 15,
  width              = 16,
  aux_not_bend       = 17,
  aux_is_rectilinear = 18,
  aux_ensure         = 19
};


template <typename edge>
inline bool is_violation(edge      edge1,
                         edge      edge2,
                         int       threshold) {
  auto [start_point1, end_point1, distance1] = edge1;
  auto [start_point2, end_point2, distance2] = edge2;
  bool is_too_close = std::abs(distance1 - distance2) < threshold ;
  bool is_projection_overlap = !(start_point1 >= end_point2 or start_point2 >= end_point1);
  return is_too_close and is_projection_overlap ;
}

inline void convert_cell_to_ref(violation_information               vio,
                         int                                 ref_x,
                         int                                 ref_y,
                         std::vector<violation_information>& vios) {
  vios.emplace_back(violation_information{
      core::edge{vio.edge1.x_startpoint + ref_x, vio.edge1.y_startpoint + ref_y,
                 vio.edge1.x_endpoint + ref_x, vio.edge1.y_endpoint + ref_y},
      core::edge{vio.edge2.x_startpoint + ref_x, vio.edge2.y_startpoint + ref_y,
                 vio.edge2.x_endpoint + ref_x, vio.edge2.y_endpoint + ref_y}});
};
inline void get_ref_vios(odrc::core::database&                             db,
                  std::map<int, std::vector<violation_information>> intra_vios,
                  std::vector<violation_information>&               vios) {
  auto& cell_refs = db.get_top_cell().cell_refs;
  for (auto num = 0UL; num < cell_refs.size(); num++) {
    for (auto& intra_vio : intra_vios) {
      if (intra_vio.second.size() != 0 and
          cell_refs.at(num).cell_name == db.cells.at(intra_vio.first).name) {
        for (const auto& vio : intra_vios.at(intra_vio.first)) {
          convert_cell_to_ref(vio, cell_refs.at(num).ref_point.x,
                              cell_refs.at(num).ref_point.y, vios);
        }
      }
    }
  }
};
inline void check_distance(std::vector<core::orthogonal_edge>  edges1,
                    std::vector<core::orthogonal_edge>  edges2,
                    int                                 threshold,
                    std::vector<violation_information>& vios) {
  for (const auto& edge1 : edges1) {
    for (const auto& edge2 : edges2) {
      auto [start_point1, end_point1, distance1] = edge1;
      auto [start_point2, end_point2, distance2] = edge2;
      if (distance1 > distance2 and
          is_violation(edge1, edge2, threshold)) {
        vios.emplace_back(violation_information{
            core::edge{start_point1, distance1, end_point1, distance1},
            core::edge{start_point2, distance2, end_point2, distance2}});
      }
    }
  }
}
void space_check_seq(odrc::core::database&               db,
                     std::vector<int>                    layers,
                     std::vector<int>                    without_layer,
                     int                                 threshold,
                     rule_type                           ruletype,
                     std::vector<violation_information>& vios);

void enclosing_check_seq(odrc::core::database&               db,
                         std::vector<int>                    layers,
                         std::vector<int>                    without_layer,
                         int                                 threshold,
                         rule_type                           ruletype,
                         std::vector<violation_information>& vios);
void area_check_seq(const odrc::core::database&         db,
                    int                                 layer,
                    int                                 threshold,
                    std::vector<violation_information>& vios);
void width_check_seq(odrc::core::database&         db,
                     int                                 layer,
                     int                                 threshold,
                     std::vector<violation_information>& vios);

}  // namespace odrc