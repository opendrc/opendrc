#pragma once

#include <odrc/algorithm/space-check.hpp>
#include <odrc/algorithm/width-check.hpp>
#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <vector>

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
      switch (rule.ruletype) {
        case rule_type::spacing_both: {
          db.update_mbr(rule.layer.front(), rule.without_layer);
          if (mod == mode::sequence) {
            space_check_seq(db, rule.layer, rule.region.first, rule.ruletype,
                            vlts);
            std::cout << vlts.size() << std::endl;
          } else if (mod == mode::parallel) {
            space_check_pal(db, rule.layer.front(), rule.layer.back(),
                            rule.region.first, vlts);
            std::cout << vlts.size() << std::endl;
          }
          break;
        }
        case rule_type::width: {
          if (mod == mode::sequence) {
            width_check_seq(db, rule.layer.front(), rule.region.first, vlts);
            std::cout << vlts.size() << std::endl;
          } else if (mod == mode::parallel) {
            width_check_pal(db, rule.layer.front(), rule.region.first, vlts);
            std::cout << vlts.size() << std::endl;
          }
        }
        default:
          break;
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

  engine& width() {
    rules.back().ruletype = rule_type::width;
    return *this;
  }

  engine& spacing() {
    rules.back().ruletype = rule_type::spacing_both;
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