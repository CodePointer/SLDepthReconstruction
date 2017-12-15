//
// Created by pointer on 17-12-15.
//

#ifndef SLDEPTHRECONSTRUCTION_VERTEX_SET_H
#define SLDEPTHRECONSTRUCTION_VERTEX_SET_H


#include <static_para.h>
#include <Eigen/Eigen>

class VertexSet {
public:
  int block_size_;
  int len_;
  // [[value], [pos_x], [pos_y], [valid]]
  Eigen::Matrix<double, Eigen::Dynamic, 1> vertex_val_;
  Eigen::Matrix<int, Eigen::Dynamic, 2> pos_;
  Eigen::Matrix<uchar, Eigen::Dynamic, 1> valid_;


  VertexSet(int block_size);
  ~VertexSet();
  bool IsVertex(int x, int y);
  Eigen::Vector4i Find4ConnectVertex(int x, int y);
  int GetVertexIdxByPos(int x, int y);
  bool GetValidStatusByPos(int x, int y);
  bool GetNeighborVertexIdxByIdx(int idx_i, const uchar dir);

};


#endif //SLDEPTHRECONSTRUCTION_VERTEX_SET_H
