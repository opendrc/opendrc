#pragma once

#include <set>
#include <vector>

#include <odrc/algorithm/parallel_mode.hpp>
#include <odrc/algorithm/sequential_mode.hpp>

#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>

namespace odrc::core {
enum class object {
  polygon,
  cell,
  both,
};
enum class sramdrc_set {
  dft              = 0,  // default
  interac_SRAMDRAC = 1,  // interact with the layer SRAMDRC
  outside_SRAMDRAC = 2,  // outside the layer SRAMDRC
};
enum class mode { sequential, parallel };
struct rule {
  int                 rule_num;
  std::vector<int>    layer;
  std::vector<int>    with_layer;
  std::vector<int>    without_layer;
  std::pair<int, int> region;
  rule_type           ruletype;
  object              obj   = object::both;
  sramdrc_set         s_set = sramdrc_set::dft;
};

class engine {
 public:
  mode                         check_mode = mode::sequential;
  std::vector<odrc::violation> vlts;
  //<rule number, polgon/cell number>
  std::vector<std::pair<int, std::pair<int, int>>> vlt_paires;
  //<rule number,<polygons/cells number pair>>

  void add_rules(std::vector<int> value) {
    std::cout << "ALL rules have been added." << std::endl;
  };

  void set_mode(mode md) { check_mode = md; };
  void check(odrc::core::database& db) {
    for (const auto& rule : rules) {
      if (check_mode == mode::sequential) {
        switch (rule.ruletype) {
          case rule_type::spacing_both: {
            space_check_seq(db, rule.layer, rule.without_layer,
                            rule.region.first, rule.ruletype, vlts);
            std::cout << vlts.size() << std::endl;
            // for (const auto& vio : vlts) {
            //   std::cout << " " << vio.distance.edge1.point1.x << " "
            //             << vio.distance.edge1.point1.y << " "
            //             << vio.distance.edge1.point2.x << " "
            //             << vio.distance.edge1.point2.y << " "
            //             << vio.distance.edge2.point1.x << " "
            //             << vio.distance.edge2.point1.y << " "
            //             << vio.distance.edge2.point2.x << " "
            //             << vio.distance.edge2.point2.y << std::endl;
            // }
            break;
          }
          case rule_type::width: {
            width_check_seq(db, rule.layer.front(), rule.region.first, vlts);
            std::cout << vlts.size() << std::endl;
            // for (const auto& vio : vlts) {
            //   std::cout << " " << vio.distance.edge1.point1.x << " "
            //             << vio.distance.edge1.point1.y << " "
            //             << vio.distance.edge1.point2.x << " "
            //             << vio.distance.edge1.point2.y << " "
            //             << vio.distance.edge2.point1.x << " "
            //             << vio.distance.edge2.point1.y << " "
            //             << vio.distance.edge2.point2.x << " "
            //             << vio.distance.edge2.point2.y << std::endl;
            // }
            break;
          }
          case rule_type::enclosure: {
            enclosure_check_seq(db, rule.layer, rule.without_layer,
                                rule.region.first, rule.ruletype, vlts);
            std::cout << vlts.size() << std::endl;
            // for (const auto& vio : vlts) {
            //   std::cout << " " << vio.distance.edge1.point1.x << " "
            //             << vio.distance.edge1.point1.y << " "
            //             << vio.distance.edge1.point2.x << " "
            //             << vio.distance.edge1.point2.y << " "
            //             << vio.distance.edge2.point1.x << " "
            //             << vio.distance.edge2.point1.y << " "
            //             << vio.distance.edge2.point2.x << " "
            //             << vio.distance.edge2.point2.y << std::endl;
            // }
            break;
          }
          case rule_type::area: {
            area_check_seq(db, rule.layer.front(), rule.region.first, vlts);
            std::cout << vlts.size() << std::endl;
            break;
          }
          default:
            break;
        }
      } else if (check_mode == mode::parallel) {
        switch (rule.ruletype) {
          case rule_type::spacing_both: {
            // space_check_par(db, rule.layer.front(), rule.layer.back(),
            //                 rule.region.first, vlts);
            break;
          }
          case rule_type::width: {
            // width_check_par(db, rule.layer.front(), rule.region.first, vlts);
            std::cout << vlts.size() << std::endl;
            break;
          }
          case rule_type::enclosure: {
            break;
          }
          case rule_type::area: {
            break;
          }
          default:
            break;
        }
      }
    }
  };

  engine& polygons() {
    rules.emplace_back();
    rules.back().rule_num = rule_num;
    rules.back().obj      = object::polygon;
    return *this;
  }

  engine& layer(int layer) {
    rules.emplace_back();
    rules.back().rule_num = rule_num;
    rules.back().layer.emplace_back(layer);
    return *this;
  }
  engine& with_layer(int layer) {
    rules.back().layer.emplace_back(layer);
    return *this;
  }
  engine& inter_layer(int layer) {
    rules.back().with_layer.emplace_back(layer);
    return *this;
  }
  engine& not_inter_layer(int layer) {
    rules.back().without_layer.emplace_back(layer);
    return *this;
  }
  engine& width() {
    rules.back().ruletype = rule_type::width;
    return *this;
  }

  engine& spacing() {
    rules.back().ruletype = rule_type::spacing_both;
    return *this;
  }
  engine& enclosure() {
    rules.back().ruletype = rule_type::enclosure;
    return *this;
  }
  engine& area() {
    rules.back().ruletype = rule_type::area;
    return *this;
  }

  int is_rectilinear() {
    rules.back().ruletype = rule_type::aux_is_rectilinear;
    return -1;
  }
  int greater_than(int min) {
    rules.back().region = std::make_pair(min, 2147483647);
    return -1;
  }

 private:
  unsigned int      rule_num = 0;
  std::vector<rule> rules;
  std::set<int>     layer_set;
};

}  // namespace odrc::core