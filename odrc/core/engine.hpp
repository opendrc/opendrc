#pragma once

#include <vector>

#include <odrc/algorithm/area-check.hpp>
#include <odrc/algorithm/enclosing-check.hpp>
#include <odrc/algorithm/space-check.hpp>
#include <odrc/algorithm/width-check.hpp>
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
enum class mode { sequence, parallel };
struct rule {
  int                 rule_num;
  std::vector<int>    layer;
  std::vector<int>    without_layer;
  std::pair<int, int> region;
  rule_type           ruletype;
  object              obj   = object::both;
  sramdrc_set         s_set = sramdrc_set::dft;
};

class engine {
 public:
  mode                            mod = mode::sequence;
  std::vector<odrc::check_result> vlts;
  //<rule number, polgon/cell number>
  std::vector<std::pair<int, std::pair<int, int>>> vlt_paires;
  //<rule number,<polygons/cells number pair>>

  void add_rules(std::vector<int> value) {
    std::cout << "ALL rules have been added." << std::endl;
  };
  void set_mode(mode md) { mod = md; };
  void check(odrc::core::database& db) {
    for (const auto& rule : rules) {
      if (mod == mode::sequence) {
        switch (rule.ruletype) {
          case rule_type::spacing_both: {
            space_check_seq(db, rule.layer, rule.without_layer,
                            rule.region.first, rule.ruletype, vlts);
            std::cout << vlts.size() << std::endl;
            // for (const auto& vio : vlts) {
            //   std::cout << "e11x: " << vio.e11x << " e11y: " << vio.e11y
            //             << " e12x: " << vio.e12x << " e12y: " << vio.e12y
            //             << " e21x: " << vio.e21x << " e21y: " << vio.e21y
            //             << " e22x: " << vio.e22x << " e22y: " << vio.e22y
            //             << std::endl;
            // }
            break;
          }
          case rule_type::width: {
            width_check_seq(db, rule.layer.front(), rule.region.first, vlts);
            std::cout << vlts.size() << std::endl;
            break;
          }
          case rule_type::enclosure: {
            enclosing_check_seq(db, rule.layer, rule.without_layer,
                                rule.region.first, rule.ruletype, vlts);
            std::cout << vlts.size() << std::endl;
            break;
          }
          case rule_type::area: {
            area_check_seq(db, rule.layer.front(), rule.region.first);
            std::cout << vlts.size() << std::endl;
            break;
          }
          default:
            break;
        }
      } else if (mod == mode::parallel) {
        switch (rule.ruletype) {
          case rule_type::spacing_both: {
            space_check_par(db, rule.layer.front(), rule.layer.back(),
                            rule.region.first, vlts);
            break;
          }
          case rule_type::width: {
            width_check_par(db, rule.layer.front(), rule.region.first, vlts);
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
  void    _schedular(){};
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
  engine& without_layer(int layer) {
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
};

}  // namespace odrc::core