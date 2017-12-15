//
// Created by pointer on 17-12-15.
//


#include "reconstructor.h"

Reconstructor::Reconstructor() {
  this->cam_set_ = nullptr;
  this->vertex_set_ = nullptr;
  this->pat_grid_ = nullptr;
}

Reconstructor::~Reconstructor() {
  if (this->cam_set_ != nullptr) {
    delete[]this->cam_set_;
    this->cam_set_ = nullptr;
  }
  if (this->vertex_set_ != nullptr) {
    delete[]this->vertex_set_;
    this->vertex_set_ = nullptr;
  }
  if (this->pat_grid_ != nullptr) {
    delete this->pat_grid_;
    this->pat_grid_ = nullptr;
  }
}

bool Reconstructor::Init() {
  bool status = true;
  // Set file path
  this->main_file_path_ = "";
  this->pattern_file_name_ = "pattern_8size2color8P0";
  this->pattern_file_suffix_ = ".png";
  this->dyna_file_path_ = "cam0/dyna/";
  this->dyna_file_name_ = "dyna_mat";
  this->dyna_file_suffix_ = ".png";
  this->pro_file_path_ = "cam0/pro/";
  this->pro_file_name_ = "xpro_mat";
  this->pro_file_suffix_ = ".txt";
  std::string epi_A_file_name_ = "EpiLine_A.txt";
  std::string epi_B_file_name_ = "EpiLine_B.txt";
  // Load Informations
  status = this->LoadDatasFromFiles();
  // Set pattern grid
  ceres::Grid2D<double, 1> * pat_grid = new ceres::Grid2D<double, 1>(
      (double*)this->pattern_.data, 0, kProHeight, 0, kProWidth);
  this->pat_grid_ = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(*pat_grid);
  return status;
}

bool Reconstructor::LoadDatasFromFiles() {
  // Calib_Set:M, D, cam_0, cam_1
  // TODO: Finish here
  // Pattern
//  this->pattern_ = cv::imread(this->main_file_path_ + this->pattern_file_name_
//                              + this->pattern_file_suffix_, cv::IMREAD_GRAYSCALE);
  this->pattern_ = LoadTxtToMat(this->main_file_path_ + this->pattern_file_name_
                                + this->pattern_file_suffix_,
                                kProHeight, kProWidth);
  // EpiLine set
  this->epi_A_mat_ = LoadTxtToMat(this->main_file_path_ + this->epi_A_file_name_,
                                  kCamHeight, kCamWidth);
  this->epi_B_mat_ = LoadTxtToMat(this->main_file_path_ + this->epi_B_file_name_,
                                  kCamHeight, kCamWidth);
  // Cam_set
  this->cam_set_ = new CamMatSet[kFrameNum];
  this->vertex_set_ = new VertexSet[kFrameNum](kGridSize);
  for (int frm_idx = 0; frm_idx < kFrameNum; frm_idx++) {
    this->cam_set_[frm_idx].img_obs = cv::imread(this->main_file_path_
                                                 + this->dyna_file_path_
                                                 + this->dyna_file_name_
                                                 + Num2Str(frm_idx)
                                                 + this->dyna_file_suffix_);
  }
  // Load first data: x_pro & depth
  this->cam_set_[0].x_pro = LoadTxtToMat(this->main_file_path_
                                       + this->pro_file_path_
                                       + this->pro_file_name_
                                       + Num2Str(0)
                                       + this->pro_file_suffix_,
                                       kCamHeight, kCamWidth);
  this->ConvXpro2Depth(&this->cam_set_[0]);
  this->SetVertexFromBefore(0);
  return true;
}

void Reconstructor::ConvXpro2Depth(CamMatSet *ptr_cam_set) {
  ptr_cam_set->depth.create(kCamHeight, kCamWidth, CV_64FC1);
  Eigen::Vector3d vec_D = this->calib_set_.D;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      int idx_k = h * kCamWidth + w;
      double x_pro = ptr_cam_set->x_pro.at<double>(h, w);
      Eigen::Vector3d vec_M = this->calib_set_.M.block<3, 1>(0, idx_k);
      ptr_cam_set->depth.at<double>(h, w) = - (vec_D(1) - vec_D(3) * x_pro)
                                            / (vec_M(1) - vec_M(3) * x_pro);
    }
  }
}

bool Reconstructor::Run() {
  bool status = true;

  for (int frm_idx = 1; frm_idx < kFrameNum; frm_idx++) {
    // Set a mask according to I_obs
    this->SetMaskMatFromIobs(frm_idx);
    // Set vertex initial value
    this->SetVertexFromBefore(frm_idx);
    // Optimization
    this->OptimizeDepthMat(frm_idx); // TODO: Add 2 Functors
    // Write result to file
    this->WriteResult(frm_idx); // TODO: Add Storage Module
  }
  return status;
}

void Reconstructor::SetMaskMatFromIobs(int frm_idx) {
  this->cam_set_[frm_idx].mask.create(kCamHeight, kCamWidth, CV_8UC1);
  // Set mask initial value by intensity
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (this->cam_set_[frm_idx].img_obs.at<uchar>(h, w) <= kMaskIntensityThred) {
        this->cam_set_[frm_idx].mask.at<uchar>(h, w) = my::INITIAL_FALSE;
      } else {
        this->cam_set_[frm_idx].mask.at<uchar>(h, w) = my::INITIAL_TRUE;
      }
    }
  }
  // FloodFill every point
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      uchar val = this->cam_set_[frm_idx].mask.at<uchar>(h, w);
      if (val == my::INITIAL_FALSE) {
        int num = FloodFill(this->cam_set_[frm_idx].mask, h, w,
                            my::MARKED, my::INITIAL_FALSE);
        if (num <= kMastMinAreaThred) {
          FloodFill(this->cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_TRUE, my::MARKED);
        } else {
          FloodFill(this->cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_FALSE, my::MARKED);
        }
      } else if (val == my::INITIAL_TRUE) {
        int num = FloodFill(this->cam_set_[frm_idx].mask, h, w,
                            my::MARKED, my::INITIAL_TRUE);
        if (num <= kMastMinAreaThred) {
          FloodFill(this->cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_FALSE, my::MARKED);
        } else {
          FloodFill(this->cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_TRUE, my::MARKED);
        }
      }
    }
  }
  // Set result with hard_mask
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (this->hard_mask_.at<uchar>(h, w) == my::VERIFIED_FALSE) {
        this->cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_FALSE;
      }
    }
  }
}

void Reconstructor::SetVertexFromBefore(int frm_idx) {
  if (frm_idx == 0) { // Have depth & mask
    for (int i = 0; i < this->vertex_set_[frm_idx].len_; i++) {
      int x = this->vertex_set_[frm_idx].pos_(i, 0);
      int y = this->vertex_set_[frm_idx].pos_(i, 1);
      if (this->cam_set_[frm_idx].mask.at<uchar>(y, x) == my::VERIFIED_TRUE) {
        this->vertex_set_[frm_idx].vertex_val_(i)
            = this->cam_set_[frm_idx].depth.at<double>(y, x);
        this->vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
      }
    }
  } else { // Not have
    // Check last frame
    for (int i = 0; i < this->vertex_set_[frm_idx].len_; i++) {
      int x = this->vertex_set_[frm_idx].pos_(i, 0);
      int y = this->vertex_set_[frm_idx].pos_(i, 1);
      if (this->cam_set_[frm_idx].mask.at<uchar>(y, x) == my::VERIFIED_TRUE) {
        if (this->vertex_set_[frm_idx - 1].valid_(i) == my::VERIFIED_TRUE) {
          this->vertex_set_[frm_idx].vertex_val_(i)
              = this->vertex_set_[frm_idx - 1].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
        } else {
          this->vertex_set_[frm_idx].valid_(i) = my::INITIAL_TRUE;
        }
      }
    }
    // Fill valid mask: for every point, find nearest point
    for (int i = 0; i < this->vertex_set_[frm_idx].len_; i++) {
      int w = this->vertex_set_[frm_idx].pos_(i, 0);
      int h = this->vertex_set_[frm_idx].pos_(i, 1);
      if (this->vertex_set_[frm_idx].valid_(i) != my::INITIAL_TRUE) {
        continue;
      }
      int rad = 1;
      bool break_flag = false;
      while (!break_flag) {
        for (int r = -rad; r <= rad; r++) {
          int w_new = w + r;
          if (this->cam_set_[frm_idx].mask.at<uchar>(h - rad, w_new)
              == my::VERIFIED_TRUE) {
            this->vertex_set_[frm_idx].vertex_val_(i)
                = this->cam_set_[frm_idx].depth.at<double>(h - rad, w_new);
            this->vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
            break_flag = true;
            break;
          }
          if (this->cam_set_[frm_idx].mask.at<uchar>(h + rad, w_new)
              == my::VERIFIED_TRUE) {
            this->vertex_set_[frm_idx].vertex_val_(i)
                = this->cam_set_[frm_idx].depth.at<double>(h + rad, w_new);
            this->vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
            break_flag = true;
            break;
          }
        }
        for (int r = -rad+1; r <= rad-1; r++) {
          int h_new = h + r;
          if (this->cam_set_[frm_idx].mask.at<uchar>(h_new, w - rad)
              == my::VERIFIED_TRUE) {
            this->vertex_set_[frm_idx].vertex_val_(i)
                = this->cam_set_[frm_idx].depth.at<double>(h_new, w - rad);
            this->vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
            break_flag = true;
            break;
          }
          if (this->cam_set_[frm_idx].mask.at<uchar>(h_new, w + rad)
              == my::VERIFIED_TRUE) {
            this->vertex_set_[frm_idx].vertex_val_(i)
                = this->cam_set_[frm_idx].depth.at<double>(h_new, w + rad);
            this->vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
            break_flag = true;
            break;
          }
        }
        rad++;
      }
    }
    // Spread: 4 & 8
    for (int i = 0; i < this->vertex_set_[frm_idx].len_; i++) {
      if (this->vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE) {
        int left_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_LEFT);
        if ((left_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(left_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(left_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(left_idx) = my::NEIGHBOR_4;
        }
        int right_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_RIGHT);
        if ((right_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(right_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(right_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(right_idx) = my::NEIGHBOR_4;
        }
        int down_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_DOWN);
        if ((down_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(down_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(down_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(down_idx) = my::NEIGHBOR_4;
        }
        int up_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_UP);
        if ((up_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(up_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(up_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(up_idx) = my::NEIGHBOR_4;
        }
      }
    }
    // Spread: 8
    for (int i = 0; i < this->vertex_set_[frm_idx].len_; i++) {
      if (this->vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE) {
        int ul_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_UP_LEFT);
        if ((ul_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(ul_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(ul_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(ul_idx) = my::NEIGHBOR_8;
        }
        int ur_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_UP_RIGHT);
        if ((ur_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(ur_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(ur_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(ur_idx) = my::NEIGHBOR_8;
        }
        int dr_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_DOWN_RIGHT);
        if ((dr_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(dr_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(dr_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(dr_idx) = my::NEIGHBOR_8;
        }
        int dl_idx = this->vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
            i, my::DIREC_DOWN_LEFT);
        if ((dl_idx > 0)
            && (this->vertex_set_[frm_idx].valid_(dl_idx) == my::VERIFIED_TRUE)) {
          this->vertex_set_[frm_idx].vertex_val_(dl_idx)
              = this->vertex_set_[frm_idx].vertex_val_(i);
          this->vertex_set_[frm_idx].valid_(dl_idx) = my::NEIGHBOR_8;
        }
      }
    }
    for (int i = 0; i < this->vertex_set_[frm_idx].len_; i++) {
      if ((this->vertex_set_[frm_idx].valid_(i) == my::NEIGHBOR_4)
          || (this->vertex_set_[frm_idx].valid_(i) == my::NEIGHBOR_8)) {
        this->vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE;
      }
    }
  }
}

bool Reconstructor::OptimizeDepthMat(int frm_idx) {
  bool status = true;
  ceres::Problem problem;

  // Add blocks
  int block_num = 0;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (this->cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE) {
        continue;
      }
      // Add Data Term
      // TODO
      block_num++;
      // Add Regular Term
      if (this->vertex_set_[frm_idx].IsVertex(w, h)) {
        // TODO
        block_num++;
      }
    }
  }

  ceres::Solver::Options options;
  options.gradient_tolerance = 1e-8;
  options.function_tolerance = 1e-8;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 60;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  // Save?
  std::cout << summary.BriefReport() << std::endl;
  return status;
}

void Reconstructor::WriteResult(int frm_idx) {

}

bool Reconstructor::Close() {
  return false;
}