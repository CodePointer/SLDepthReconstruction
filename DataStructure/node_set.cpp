//
// Created by pointer on 18-4-18.
//

#include "node_set.h"

NodeSet::NodeSet() {
  block_size_ = kNodeBlockSize;
  block_height_ = kCamHeight / block_size_;
  block_width_ = kCamWidth / block_size_;
  len_ = block_height_ * block_width_;
  val_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(len_, 1);
  pos_ = Eigen::Matrix<int, Eigen::Dynamic, 2>::Zero(len_, 2);
  valid_ = Eigen::Matrix<uchar, Eigen::Dynamic, 1>::Zero(len_, 1);
  for (int h = 0; h < block_height_; h++) {
    for (int w = 0; w < block_width_; w++) {
      int idx_i = h * block_width_ + w;
      val_(idx_i, 0) = -1.0;
      valid_(idx_i, 0) = my::VERIFIED_FALSE;
    }
  }
}

NodeSet::~NodeSet() = default;

void NodeSet::Clear() {
  val_.resize(0, 1);
  pos_.resize(0, 2);
  valid_.resize(0, 1);
}

bool NodeSet::IsNode(int x, int y) {
  int h, w;
  GetNodeCoordByPos(x, y, &h, &w);
  int idx = h * block_width_ + w;
  int pos_x = pos_(idx, 0);
  int pos_y = pos_(idx, 1);
  return (pos_x == x) && (pos_y == y);
}

void NodeSet::GetNodeCoordByPos(int x, int y, int * h, int * w) {
  if (h != nullptr) {
    *h = (int)floor(y / block_size_);
  }
  if (w != nullptr) {
    *w = (int)floor(x / block_size_);
  }
}

void NodeSet::GetNodeCoordByIdx(int idx, int * h, int * w) {
  if (h != nullptr) {
    *h = idx / block_width_;
  }
  if (w != nullptr) {
    *w = idx % block_width_;
  }
}

bool NodeSet::SetNodePos(int idx, int x, int y) {
  int h, w;
  GetNodeCoordByIdx(idx, &h, &w);
  if (x < w * block_size_ || x >= (w+1) * block_size_) {
    LOG(ERROR) << "x is invalid: x,y=(" << x << "," << y << ")";
    return false;
  }
  if (y < h * block_size_ || y >= (h+1) * block_size_) {
    LOG(ERROR) << "y is invalid: x,y=(" << x << "," << y << ")";
    return false;
  }
  pos_(idx, 0) = x;
  pos_(idx, 1) = y;
  return true;
}

Eigen::Matrix<double, Eigen::Dynamic, 2> NodeSet::FindkNearestNodes(
    int x, int y, int k) {
  // Fill the nbr_set: [(idx_0; dis_0), (idx_1; dis_1), ... ]
  Eigen::Matrix<double, Eigen::Dynamic, 2> nbr_set
      = Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(k, 2);

  // Search possible vertex
  std::vector<std::pair<int ,double>> possible_set;
  int h_cen, w_cen;
  GetNodeCoordByPos(x, y, &h_cen, &w_cen);
  int h_s, w_s;
  int rad = 0;
  int rad_thred = 1000;
  h_s = h_cen; w_s = w_cen;
  bool search_flag = true;
  while (search_flag) {
    // Check h_s, w_s status
    if (h_s >= 0 && h_s < block_height_ && w_s >= 0 && w_s < block_width_) {
      int idx_s = h_s * block_width_ + w_s;
      if (valid_(idx_s, 0) == my::VERIFIED_TRUE) {
        int pos_x = pos_(idx_s, 0);
        int pos_y = pos_(idx_s, 1);
        double dist = sqrt((pos_x-x)*(pos_x-x) + (pos_y-y)*(pos_y-y));
        possible_set.emplace_back(std::pair<int, double>(idx_s, dist));
      }
    }
    // Set Next h_s & w_s
    if (h_s == h_cen - rad && w_s < w_cen + rad) {
      w_s++; // up
    } else if (w_s == w_cen + rad && h_s < h_cen + rad) {
      h_s++; // right
    } else if (h_s == h_cen + rad && w_s > w_cen - rad) {
      w_s--; // down
    } else if (w_s == w_cen - rad && h_s > h_cen - rad + 1) {
      h_s--; // left
    } else if ((h_s == h_cen - rad + 1 && w_s == w_cen - rad)
               || (h_s == h_cen && w_s == w_cen)) {
      // Check
      if (possible_set.size() >= k || rad > rad_thred)
        search_flag = false;
      rad++;
      h_s = h_cen - rad;
      w_s = w_cen - rad;
    } else {
      LOG(ERROR) << "Logic problem.";
    }
  }

  // Sort by dist
  for (int i = 0; i < k; i++) {
    for (int j = (int)possible_set.size() - 2; j >= 0; j--) {
      std::pair<int, double> node_back = possible_set[j + 1];
      std::pair<int, double> node_frnt = possible_set[j];
      if (node_back.second < node_frnt.second) {
        possible_set[j + 1] = node_frnt;
        possible_set[j] = node_back;
      }
    }
    nbr_set(i, 0) = possible_set[i].first;
    nbr_set(i, 1) = possible_set[i].second;
  }

  return nbr_set;
};

bool NodeSet::WriteToFile(std::string file_name) {
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    LOG(ERROR) << "WriteToFile error, file_name=" + file_name;
    return false;
  }
  for (int i = 0; i < len_; i++) {
    file << (int)valid_(i, 0) << " ";
    file << (double)val_(i, 0) << " ";
    file << pos_(i, 0) << " " << pos_(i, 1) << std::endl;
  }
  file.close();
  return true;
}