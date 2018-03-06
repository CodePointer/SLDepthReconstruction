//
// Created by pointer on 17-12-15.
//

#ifndef SLDEPTHRECONSTRUCTION_VERTEX_SET_H
#define SLDEPTHRECONSTRUCTION_VERTEX_SET_H


#include "static_para.h"
#include "global_fun.h"
#include <Eigen/Eigen>
#include <vector>

class VertexSet {
public:
  int block_size_;
  int block_height_;
  int block_width_;
  int len_;
  // [[value], [pos_x], [pos_y], [valid]]
  Eigen::Matrix<double, Eigen::Dynamic, 1> vertex_val_;
  Eigen::Matrix<int, Eigen::Dynamic, 2> pos_;
  Eigen::Matrix<uchar, Eigen::Dynamic, 1> valid_;


  VertexSet();
  ~VertexSet();
  bool IsVertex(int x, int y);
  Eigen::Matrix<double, Eigen::Dynamic, 2> FindkNearestVertex(int x, int y, int k);
  Eigen::Vector4i Find4ConnectVertex(int x, int y);
  int GetVertexIdxByPos(int x, int y);
  uchar GetValidStatusByPos(int x, int y);
  int GetNeighborVertexIdxByIdx(int idx_i, const uchar dir);

};


#endif //SLDEPTHRECONSTRUCTION_VERTEX_SET_H
