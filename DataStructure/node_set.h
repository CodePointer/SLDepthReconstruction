//
// Created by pointer on 18-4-18.
//

#ifndef SLDEPTHRECONSTRUCTION_NODE_SET_H
#define SLDEPTHRECONSTRUCTION_NODE_SET_H

#include "static_para.h"
#include "global_fun.h"
#include "intensity_slot.h"
#include <Eigen/Eigen>

class NodeSet {
public:
  int block_size_;
  int block_height_;
  int block_width_;
  int len_;

  Eigen::Matrix<double, Eigen::Dynamic, 1> val_;
  Eigen::Matrix<double, Eigen::Dynamic, 2> bound_;
  Eigen::Matrix<int, Eigen::Dynamic, 2> pos_;
  Eigen::Matrix<uchar, Eigen::Dynamic, 1> valid_;

  std::vector<cv::Point3i> mesh_;

  NodeSet();
  ~NodeSet();
  void Clear();
  bool CreateMeshSet();
  bool FillMatWithMeshIdx(CamMatSet* cam_set);

  bool IsNode(int x, int y);
  void GetNodeCoordByPos(int x, int y, int * h, int * w);
  void GetNodeCoordByIdx(int idx, int * h, int * w);
  int GetIdxByNodeCoord(int h, int w, bool valid_flag = true);
  void GetTriVertexIdx(int x, int y, std::vector<int>* res);
  void GetNearestkNodesIdx(int x, int y, std::vector<int>* res);
  bool SetNodePos(int idx, int x, int y);
  Eigen::Matrix<double, Eigen::Dynamic, 2> FindkNearestNodes(int x, int y, int k);

  bool WriteToFile(std::string file_name);
};


#endif //SLDEPTHRECONSTRUCTION_NODE_SET_H
