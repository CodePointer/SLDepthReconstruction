//
// Created by pointer on 17-12-15.
//

#include "vertex_set.h"

VertexSet::VertexSet(int block_size) {
  this->block_size_ = block_size;
  int block_height = int(floor(kCamHeight / block_size));
  int block_width = int(floor(kCamWidth / block_size));
  this->len_ = block_height * block_width;
  this->vertex_val_
      = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(this->len_, 1);
  this->pos_
      = Eigen::Matrix<int, Eigen::Dynamic, 2>::Zero(this->len_, 2);
  this->valid_
      = Eigen::Matrix<uchar, Eigen::Dynamic, 1>::Zero(this->len_, 1);
  for (int h = 0; h < block_height; h++) {
    for (int w = 0; w < block_width; w++) {
      int y = h * this->block_size_;
      int x = w * this->block_size_;
      int idx_i = h*block_width + w;
      this->vertex_val_(idx_i) = -1.0;
      this->pos_(idx_i, 0) = x;
      this->pos_(idx_i, 1) = y;
      this->valid_(idx_i) = 0;
    }
  }
}

VertexSet::~VertexSet() = default;

bool VertexSet::IsVertex(int x, int y) {
  return (x % this->block_size_ == 0) && (y % this->block_size_ == 0);
}

Eigen::Vector4i VertexSet::Find4ConnectVertex(int x, int y) {
  Eigen::Vector4i result = Eigen::Vector4i::Zero();
  int h_up = int(floor(double(y) / this->block_size_) * this->block_size_);
  int h_dn = h_up + this->block_size_;
  int w_lf = int(floor(double(x) / this->block_size_) * this->block_size_);
  int w_rt = w_lf + this->block_size_;
  // UpLf,UpRt,DnRt,DnLf
  if ((h_up >= 0) && (w_lf >= 0)) {
    result(0) = this->GetVertexIdxByPos(w_lf, h_up);
  } else {
    result(0) = -1;
  }
  if ((h_up >= 0) && (w_rt < kCamWidth)) {
    result(1) = this->GetVertexIdxByPos(w_rt, h_up);
  } else {
    result(1) = -1;
  }
  if ((h_dn < kCamHeight) && (w_rt < kCamWidth)) {
    result(2) = this->GetVertexIdxByPos(w_rt, h_dn);
  } else {
    result(2) = -1;
  }
  if ((h_dn < kCamHeight) && (w_lf >= 0)) {
    result(3) = this->GetVertexIdxByPos(w_lf, h_dn);
  } else {
    result(3) = -1;
  }
}

int VertexSet::GetVertexIdxByPos(int x, int y) {
  if ((x >= 0) && (x < kCamWidth) && (y >= 0) && (y < kCamHeight)
      && (x % this->block_size_ == 0) && (y % this->block_size_ == 0)) {
    int block_width = int(floor(kCamWidth / this->block_size_));
    int h = y / 15;
    int w = x / 15;
    return h * block_width + w;
  }
  return -1;
}

bool VertexSet::GetValidStatusByPos(int x, int y) {
  if ((x >= 0) && (x < kCamWidth) && (y >= 0) && (y < kCamHeight)
      && (x % this->block_size_ == 0) && (y % this->block_size_ == 0)) {
    return this->valid_(this->GetVertexIdxByPos(x, y));
  }
  return false;
}

int VertexSet::GetNeighborVertexIdxByIdx(int idx_i, const uchar dir) {
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


