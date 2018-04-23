//
// Created by pointer on 18-4-18.
//

#ifndef SLDEPTHRECONSTRUCTION_NODE_SET_H
#define SLDEPTHRECONSTRUCTION_NODE_SET_H

#include "static_para.h"
#include "global_fun.h"
#include <Eigen/Eigen>

class NodeSet {
public:
  int block_size_;
  int block_height_;
  int block_width_;
  int len_;

  Eigen::Matrix<double, Eigen::Dynamic, 1> val_;
  Eigen::Matrix<int, Eigen::Dynamic, 2> pos_;
  Eigen::Matrix<uchar, Eigen::Dynamic, 1> valid_;

  NodeSet();
  ~NodeSet();
  void Clear();
  bool IsNode(int x, int y);
  void GetNodeCoordByPos(int x, int y, int * h, int * w);
  void GetNodeCoordByIdx(int idx, int * h, int * w);
  bool SetNodePos(int idx, int x, int y);
  Eigen::Matrix<double, Eigen::Dynamic, 2> FindkNearestNodes(int x, int y, int k);
  bool WriteToFile(std::string file_name);
};


#endif //SLDEPTHRECONSTRUCTION_NODE_SET_H
