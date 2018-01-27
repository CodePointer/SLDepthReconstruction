//
// Created by pointer on 17-12-15.
//

#include "vertex_set.h"

VertexSet::VertexSet() {
  block_size_ = kGridSize;
  block_height_ = int(floor((kCamHeight - 1) / block_size_) + 1);
  block_width_ = int(floor((kCamWidth - 1) / block_size_) + 1);
  len_ = block_height_ * block_width_;
  vertex_val_ = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(len_, 1);
  pos_ = Eigen::Matrix<int, Eigen::Dynamic, 2>::Zero(len_, 2);
  valid_ = Eigen::Matrix<uchar, Eigen::Dynamic, 1>::Zero(len_, 1);
  for (int h = 0; h < block_height_; h++) {
    for (int w = 0; w < block_width_; w++) {
      int y = h * block_size_;
      int x = w * block_size_;
      int idx_i = h*block_width_ + w;
      vertex_val_(idx_i) = -1.0;
      pos_(idx_i, 0) = x;
      pos_(idx_i, 1) = y;
      valid_(idx_i) = my::VERIFIED_FALSE;
    }
  }
}

VertexSet::~VertexSet() = default;

bool VertexSet::IsVertex(int x, int y) {
  return (x % block_size_ == 0) && (y % block_size_ == 0);
}

Eigen::Vector4i VertexSet::Find4ConnectVertex(int x, int y) {
  Eigen::Vector4i result = Eigen::Vector4i::Zero();
  int h_up = int(floor(double(y) / block_size_) * block_size_);
  int h_dn = h_up + this->block_size_;
  int w_lf = int(floor(double(x) / block_size_) * block_size_);
  int w_rt = w_lf + block_size_;
  // UpLf,UpRt,DnRt,DnLf
  if ((h_up >= 0) && (w_lf >= 0)) {
    result(0) = GetVertexIdxByPos(w_lf, h_up);
  } else {
    result(0) = -1;
  }
  if ((h_up >= 0) && (w_rt < kCamWidth)) {
    result(1) = GetVertexIdxByPos(w_rt, h_up);
  } else {
    result(1) = -1;
  }
  if ((h_dn < kCamHeight) && (w_rt < kCamWidth)) {
    result(2) = GetVertexIdxByPos(w_rt, h_dn);
  } else {
    result(2) = -1;
  }
  if ((h_dn < kCamHeight) && (w_lf >= 0)) {
    result(3) = GetVertexIdxByPos(w_lf, h_dn);
  } else {
    result(3) = -1;
  }
  return result;
}

int VertexSet::GetVertexIdxByPos(int x, int y) {
  if ((x >= 0) && (x < kCamWidth) && (y >= 0) && (y < kCamHeight)
      && (x % block_size_ == 0) && (y % block_size_ == 0)) {
    int h = y / 15;
    int w = x / 15;
    return h * block_width_ + w;
  }
  return -1;
}

bool VertexSet::GetValidStatusByPos(int x, int y) {
  if ((x >= 0) && (x < kCamWidth) && (y >= 0) && (y < kCamHeight)
      && (x % block_size_ == 0) && (y % block_size_ == 0)) {
    return valid_(GetVertexIdxByPos(x, y));
  }
  return false;
}

int VertexSet::GetNeighborVertexIdxByIdx(int idx_i, const uchar dir) {
  if (idx_i >= len_) {
    ErrorThrow("VertexSet::GetNeighborVertexIdxByIdx, idx_i=" + Num2Str(idx_i));
    return -1;
  }
  int x = this->pos_(idx_i, 0);
  int y = this->pos_(idx_i, 1);
  int x_left = x - this->block_size_;
  int x_right = x + this->block_size_;
  int y_up = y - this->block_size_;
  int y_down = y + this->block_size_;
  switch (dir) {
    case my::DIREC_UP_LEFT:
      return this->GetVertexIdxByPos(x_left, y_up);
    case my::DIREC_UP:
      return this->GetVertexIdxByPos(x, y_up);
    case my::DIREC_UP_RIGHT:
      return this->GetVertexIdxByPos(x_right, y_up);
    case my::DIREC_RIGHT:
      return this->GetVertexIdxByPos(x_right, y);
    case my::DIREC_DOWN_RIGHT:
      return this->GetVertexIdxByPos(x_right, y_down);
    case my::DIREC_DOWN:
      return this->GetVertexIdxByPos(x, y_down);
    case my::DIREC_DOWN_LEFT:
      return this->GetVertexIdxByPos(x_left, y_down);
    case my::DIREC_LEFT:
      return this->GetVertexIdxByPos(x_left, y);
    default:
      return false;
  }
}


