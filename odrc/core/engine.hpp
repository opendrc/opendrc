#pragma once

#include <iostream>
#include <odrc/core/cell.hpp>
#include <odrc/core/database.hpp>
#include <vector>

namespace odrc::core {
enum class object {
  both   = 0,  // default,both horizontal and vertical edges
  h_edge = 1,  // only horizontal edges
  v_edge = 2,  // only vertical edges
  corner = 3,  // corner-corner
  center = 4,  // center-center
  tip    = 5,  // tip-tip
  lup    = 6,  // latch-upobject
};

enum class sramdrc_set {
  dft              = 0,  // default
  interac_SRAMDRAC = 1,  // interact with the layer SRAMDRC
  outside_SRAMDRAC = 2,  // outside the layer SRAMDRC
};

class engine {
 public:
  odrc::core::database             database;
  std::vector<std::pair<int, int>> vlts;
  //<rule number, polgon/cell number>
  std::vector<std::pair<int, std::pair<int, int>>> vlt_paires;
  //<rule number,<number pair>>

  engine(odrc::core::database& db) { database = db; };

  void rectilinear() { std::cout << "success!" << std::endl; };
  // rectilinear check for all polygons

  void not_bend(int layer);
  void in_grid(int layer, int value);
  void not_notchy(int layer, object edge = object::both);
  void rely(int layer, int aux_layer);
  // some layers may not exist without other layers
  void equal_width(int layer, int aux_layer);
  // i.e. V1 must exactly be the same width as M2 along the
  // direction perpendicular to the M2 length

  void coincide(int layer, int aux_layer, object edge = object::both);
  // some edges of polygons in th layer should be
  // coincided with something(i.e. grid) in aux_layer
  void not_coincide(int layer, int aux_layer, object edge = object::both);
  // some edges of polygons in th layer should be
  // coincided with something(i.e. grid) in aux_layer

  void overlapping(int layer, int aux_layer);
  void not_overlapping(int         layer,
                       int         aux_layer,
                       sramdrc_set sramdrc = sramdrc_set::dft);

  void enclosed(int layer, int aux_layer);
  // the polygon in layer should be enclosed by aux_layer
  void enclosed(int layer, int aux_layer1, int aux_layer2);
  // the polygon in layer should be enclosed by
  // aux_layer1 or aux_layer2

  // without_layer,object, sramdrc, along_layer and others are the constraints
  // for target objects
  void area(int         layer,
            int         max,
            int         aux_layer     = -1,
            int         without_layer = -1,
            sramdrc_set sramdrc       = sramdrc_set::dft);
  // the overlapping area between layer and aux_layer (if there is not
  // aux_layer, it refers to the area itself) should be less than max

  void width(int                 layer,
             std::pair<int, int> range,
             object              edge        = object::both,
             sramdrc_set         sramdrc     = sramdrc_set::dft,
             int                 along_layer = -1);
  // the width of edge should meet range constraints

  void spacing(int                 layer,
               std::pair<int, int> range,
               int                 aux_layer = -1,
               object              edge      = object::both,
               sramdrc_set         sramdrc   = sramdrc_set::dft);
  // the spacing between layer and aux_layer (if there is not aux_layer, it
  // refers to the spacing in same layer) should meet range constraints

  void spacing(int                 layer,
               std::pair<int, int> range,
               object              edge       = object::both,
               sramdrc_set         sramdrc    = sramdrc_set::dft,
               int                 num_endcup = -1);
  void spacing(int layer, std::pair<int, int> edge, std::pair<int, int> range);
  // at least one vertical edge of two polgons should meet range constraints
  // edge
  void spacing(int                 layer,
               std::pair<int, int> edge1,
               std::pair<int, int> edge2,
               std::pair<int, int> range);
  // vertical edge of two polgons should meet range constraints edge1 and edge2
  // respectively

  void extension(int                 layer,
                 int                 aux_layer,
                 std::pair<int, int> range,
                 int                 without_layer = -1,
                 object              edge          = object::both,
                 sramdrc_set         sramdrc       = sramdrc_set::dft);
  void enclosure(int layer,
                 int aux_layer,
                 std::pair<int, int>,
                 object      edge    = object::both,
                 sramdrc_set sramdrc = sramdrc_set::dft);
  void overlap(int         layer,
               int         aux_layer,
               int         min,
               object      edge    = object::both,
               sramdrc_set sramdrc = sramdrc_set::dft);

 private:
  std::vector<int> selected_objects1;
  std::vector<int> selected_objects2;
};

}  // namespace odrc::core