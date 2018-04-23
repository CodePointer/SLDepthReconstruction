//
// Created by pointer on 17-12-15.
//


#include <spatial_constraint.h>
#include <temporal_constraint.h>
#include <shade_data_constraint.h>
#include <intensity_slot.h>
#include "reconstructor.h"

Reconstructor::Reconstructor() {
  cam_set_ = nullptr;
  cam_slots_ = nullptr;
  node_set_ = nullptr;
  vertex_set_ = nullptr;
  pat_grid_ = nullptr;
}

Reconstructor::~Reconstructor() {
  if (cam_set_ != nullptr) {
    delete[]cam_set_;
    cam_set_ = nullptr;
  }
  if (cam_slots_ != nullptr) {
    delete cam_slots_;
    cam_slots_ = nullptr;
  }
  if (node_set_ != nullptr) {
    delete[]node_set_;
    node_set_ = nullptr;
  }
  if (vertex_set_ != nullptr) {
    delete[]vertex_set_;
    vertex_set_ = nullptr;
  }
  if (pat_grid_ != nullptr) {
    delete pat_grid_;
    pat_grid_ = nullptr;
  }
}

bool Reconstructor::Init() {
  bool status = true;

  LOG(INFO) << "Reconstructor::Init start.";
  gettimeofday(&g_time_last, nullptr);

  // Set file path
  main_file_path_ = "/home/pointer/CLionProjects/Data/20180313/Hand_2Color_2/";
  pattern_file_name_ = "pattern_gauss";
  pattern_file_suffix_ = ".txt";
  dyna_file_path_ = "cam_0/dyna/";
  dyna_file_name_ = "dyna_mat";
  dyna_file_suffix_ = ".png";
  pro_file_path_ = "cam_0/pro/";
  pro_file_name_ = "xpro_mat";
  pro_file_suffix_ = ".txt";
  epi_A_file_name_ = "EpiMatA.txt";
  epi_B_file_name_ = "EpiMatB.txt";
  cam_matrix_name_ = "cam_matrix.txt";
  pro_matrix_name_ = "pro_matrix.txt";
  rots_name_ = "rots.txt";
  trans_name_ = "trans.txt";
  light_name_ = "light_vec.txt";
//  hard_mask_file_name_ = "cam0_pro/hard_mask.png";
  // Set output file path
  output_file_path_ = main_file_path_ + "result/";
  depth_file_path_ = "";
  depth_file_name_ = "depth";

  // Load Informations
  status = LoadDatasFromFiles();
  inten_thred_ = 125.0;
  // Set pattern grid

//  cv::namedWindow("pattern");
//  cv::imshow("pattern", pattern_ / 255.0);
//  cv::waitKey(0);
//  cv::destroyWindow("pattern");
  LOG(INFO) << "Reconstructor::Init finished.";
  return status;
}

bool Reconstructor::LoadDatasFromFiles() {
  LOG(INFO) << "Loading data from harddisk.";
  // Calib_Set:M, D, cam_0, cam_1
  LoadTxtToEigen(main_file_path_ + cam_matrix_name_, 3, 3, calib_set_.cam.data());
  LoadTxtToEigen(main_file_path_ + pro_matrix_name_, 3, 3, calib_set_.pro.data());
  LoadTxtToEigen(main_file_path_ + rots_name_, 3, 3, calib_set_.R.data());
  LoadTxtToEigen(main_file_path_ + trans_name_, 3, 1, calib_set_.t.data());
  LoadTxtToEigen(main_file_path_ + light_name_, 3, 1, calib_set_.light_vec_.data());
  calib_set_.cam_mat.block(0, 0, 3, 3) = calib_set_.cam;
  calib_set_.cam_mat.block(0, 3, 3, 1) = Eigen::Matrix<double, 3, 1>::Zero();
//  std::cout << calib_set_.cam_mat << std::endl;
  Eigen::Matrix<double, 3, 4> tmp;
//  tmp.block(0, 0, 3, 3) = calib_set_.R.transpose();
//  tmp.block(0, 3, 3, 1) = -calib_set_.t;
  tmp.block(0, 0, 3, 3) = calib_set_.R;
  tmp.block(0, 3, 3, 1) = calib_set_.t;
//  std::cout << tmp << std::endl;
  calib_set_.pro_mat = calib_set_.pro * tmp;
//  std::cout << calib_set_.pro_mat << std::endl;

  calib_set_.D = calib_set_.pro_mat.block<3, 1>(0, 3);
  calib_set_.M = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, kCamVecSize);
  double dx = calib_set_.cam_mat(0, 2);
  double dy = calib_set_.cam_mat(1, 2);
  double fx = calib_set_.cam_mat(0, 0);
  double fy = calib_set_.cam_mat(1, 1);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      int idx_k = h*kCamWidth + w;
      Eigen::Matrix<double, 3, 1> tmp_vec;
      tmp_vec << (w - dx)/fx, (h - dy)/fy, 1.0;
      Eigen::Vector3d tmp_M;
      tmp_M = calib_set_.pro_mat.block<3, 3>(0, 0) * tmp_vec;
      calib_set_.M.block(0, idx_k, 3, 1) = tmp_M;
    }
  }

  // Pattern
//  pattern_ = LoadTxtToMat(main_file_path_ + pattern_file_name_
//                                + pattern_file_suffix_,
//                                kProHeight, kProWidth);
  cv::Mat raw_pattern = cv::imread(main_file_path_ + "pattern_3size2color0.png", CV_LOAD_IMAGE_GRAYSCALE);
  pattern_.create(kProHeight, kProWidth, CV_64FC1);
  for (int h = 0; h < kProHeight; h++) {
    for (int w = 0; w < kProWidth; w++) {
      uchar raw_val = raw_pattern.at<uchar>(h, w);
      pattern_.at<double>(h, w) = (double)raw_val / 255.0 * 150 + 50;
    }
  }
  ceres::Grid2D<double, 1> * pat_grid = new ceres::Grid2D<double, 1>(
      (double*)pattern_.data, 0, kProHeight, 0, kProWidth);
  pat_grid_ = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(*pat_grid);

  // EpiLine set
  epi_A_mat_ = LoadTxtToMat(main_file_path_ + epi_A_file_name_,
                                  kCamHeight, kCamWidth);
  epi_B_mat_ = LoadTxtToMat(main_file_path_ + epi_B_file_name_,
                                  kCamHeight, kCamWidth);
  hard_mask_.create(kCamHeight, kCamWidth, CV_8UC1);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      double epi_A_val = epi_A_mat_.at<double>(h, w);
      double epi_B_val = epi_B_mat_.at<double>(h, w);
      if (epi_A_val == 0 && epi_B_val == 0) {
        hard_mask_.at<uchar>(h, w) = my::VERIFIED_FALSE;
      } else {
        hard_mask_.at<uchar>(h, w) = my::VERIFIED_TRUE;
      }
    }
  }

  // Cam_set
  cam_set_ = new CamMatSet[kFrameNum];
  cam_slots_ = new CamSlotsMat;
  node_set_ = new NodeSet[kFrameNum];
  vertex_set_ = new VertexSet[kFrameNum];
  for (int frm_idx = 0; frm_idx < kFrameNum; frm_idx++) {
    cam_set_[frm_idx].img_obs = cv::imread(main_file_path_
                                           + dyna_file_path_
                                           + dyna_file_name_
                                           + Num2Str(frm_idx)
                                           + dyna_file_suffix_,
                                           cv::IMREAD_GRAYSCALE);
  }
  // Load first data: x_pro & depth
  cam_set_[0].x_pro = LoadTxtToMat(main_file_path_
                                       + pro_file_path_
                                       + pro_file_name_
                                       + Num2Str(0)
                                       + pro_file_suffix_,
                                       kCamHeight, kCamWidth);
  LOG(INFO) << "Loading data finished.";
  return true;
}

void Reconstructor::ConvXpro2Depth(CamMatSet *ptr_cam_set) {
  LOG(INFO) << "Start: ConvXpro2Depth()";
  ptr_cam_set->depth.create(kCamHeight, kCamWidth, CV_64FC1);
  Eigen::Vector3d vec_D = calib_set_.D;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (ptr_cam_set->mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      double x_pro = ptr_cam_set->x_pro.at<double>(h, w);
      if (x_pro < 0) {
        if ((ptr_cam_set->x_pro.at<double>(h, w - 1) > 0)
            && (ptr_cam_set->x_pro.at<double>(h, w + 1) > 0)) {
          x_pro = 0.5 * (ptr_cam_set->x_pro.at<double>(h, w - 1)
                         + ptr_cam_set->x_pro.at<double>(h, w + 1));
//        } else if (ptr_cam_set->x_pro.at<double>(h, w - 1) > 0) {
//          x_pro = ptr_cam_set->x_pro.at<double>(h, w - 1);
//        } else if (ptr_cam_set->x_pro.at<double>(h, w + 1) > 0) {
//          x_pro = ptr_cam_set->x_pro.at<double>(h, w + 1);
//        } else {
//          ptr_cam_set->mask.at<uchar>(h, w) = my::MARKED;
//          continue;
//        }
        } else {
          ptr_cam_set->mask.at<uchar>(h, w) = my::VERIFIED_FALSE;
          continue;
        }
      }
      double depth = GetDepthFromXpro(x_pro, h, w, &calib_set_);
      if ((depth < kDepthMin) || (depth > kDepthMax)) {
        ptr_cam_set->depth.at<double>(h, w) = -1;
        ptr_cam_set->mask.at<uchar>(h, w) = my::MARKED;
      } else {
        ptr_cam_set->depth.at<double>(h, w) = depth;
      }
    }
  }
  // Interpolation
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (ptr_cam_set->mask.at<uchar>(h, w) == my::MARKED) {
        int search_rad = 5;
        double sum_val = 0;
        int sum_num = 0;
        for (int d_h = -search_rad; d_h <= search_rad; d_h++) {
          for (int d_w = -search_rad; d_w <= search_rad; d_w++) {
            if (ptr_cam_set->mask.at<uchar>(h+d_h, w+d_w) == my::VERIFIED_TRUE) {
              sum_num++;
              sum_val += ptr_cam_set->depth.at<double>(h+d_h, w+d_w);
            }
          }
        }
        ptr_cam_set->depth.at<double>(h, w) = sum_val / double(sum_num);
        ptr_cam_set->mask.at<uchar>(h, w) = my::VERIFIED_TRUE;
      }
    }
  }
//  cv::namedWindow("test");
//  cv::imshow("test", (ptr_cam_set->depth/100));
//  cv::waitKey(0);
  LOG(INFO) << "End: ConvXpro2Depth()";
}

bool Reconstructor::Run() {
  bool status = true;

  LOG(INFO) << "Reconstruction process start.";

  // First frame (frame 0):
  LOG(INFO) << "First frame process start.";
  SetMaskMatFromIobs(0);
  GenerateIntensitySlots(0);
  ConvXpro2Depth(&cam_set_[0]);
  SetFirstFrameVertex(0);
  FillShadeMatFromVertex(0);

  GenerateIestFromDepth(0);
  OptimizeShadingMat(0);
  FillShadeMatFromVertex(0);

  RecoIntensityClass(0);
  RefineInitialDepthVal(0);
  SetNodeFromDepthVal(0);
  SetDepthValFromNode(0);

  GenerateIestFromDepth(0);
  LOG(INFO) << "First frame process finished.";

  // Further frame:
  for (int frm_idx = 1; (frm_idx < kFrameNum) && status; frm_idx++) {
    LOG(INFO) << "Frame (" << frm_idx << ") processing";
    std::cout << "Frame (" << frm_idx << "):" << std::endl;

    SetMaskMatFromIobs(frm_idx);
    GenerateIntensitySlots(frm_idx);

    PredictInitialDepthVal(frm_idx);

    PredictInitialShadeVertex(frm_idx);
    FillShadeMatFromVertex(frm_idx);

    int k = 1;
    while (k-- > 0) {
      RecoIntensityClass(frm_idx);
      RefineInitialDepthVal(frm_idx);

      ShowMat<double>(&(cam_set_[frm_idx].depth), "depth1", 500, 25.0, 35.0);

      SetNodeFromDepthVal(frm_idx);
      LOG(INFO) << "Begin opt depth.";
      OptimizeDepthNode(frm_idx);
      LOG(INFO) << "Finish opt depth.";
      SetDepthValFromNode(frm_idx);

      GenerateIestFromDepth(frm_idx);

      ShowMat<double>(&(cam_set_[frm_idx].depth), "depth2", 500, 25.0, 35.0);

      LOG(INFO) << "Begin opt shading.";
      OptimizeShadingMat(frm_idx);
      LOG(INFO) << "Finish opt shading.";
      FillShadeMatFromVertex(frm_idx);

      GenerateIestFromDepth(frm_idx);
    }

    if (frm_idx - kTemporalWindowSize >= 1) {
      OutputResult(frm_idx - kTemporalWindowSize - 1);
      ReleaseSpace(frm_idx - kTemporalWindowSize - 1);
    }
    LOG(INFO) << "Frame " << frm_idx << "finished.";
  }
  return status;
}

void Reconstructor::SetMaskMatFromXpro(int frm_idx) {
  cam_set_[frm_idx].mask.create(kCamHeight, kCamWidth, CV_8UC1);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      double B = epi_B_mat_.at<double>(h, w);
      if (B == 0) {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_FALSE;
        continue;
      }
      if (cam_set_[frm_idx].x_pro.at<double>(h, w) <= 0) {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_FALSE;
      } else {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_TRUE;
      }
    }
  }
}

// Set first frame's vertex_set
void Reconstructor::SetFirstFrameVertex(int frm_idx) {
  LOG(INFO) << "Start: SetFirstFrameVertex(" << frm_idx << ");";
  if (frm_idx != 0) {
    LOG(ERROR) << "frm_idx != 0 : " << frm_idx;
    return;
  }
  cv::Mat gauss_depth;
  cv::GaussianBlur(cam_set_[frm_idx].depth, gauss_depth, cv::Size(3, 3), 3.0);
  double fx = calib_set_.cam_mat(0, 0);
  double fy = calib_set_.cam_mat(1, 1);
  double dx = calib_set_.cam_mat(0, 2);
  double dy = calib_set_.cam_mat(1, 2);
  Eigen::Vector3d D = calib_set_.D;
  for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
    int x = vertex_set_[frm_idx].pos_(i, 0);
    int y = vertex_set_[frm_idx].pos_(i, 1);
    if (cam_set_[frm_idx].mask.at<uchar>(y + 1, x + 1) == my::VERIFIED_TRUE
        && cam_set_[frm_idx].mask.at<uchar>(y + 1, x) == my::VERIFIED_TRUE
        && cam_set_[frm_idx].mask.at<uchar>(y, x + 1) == my::VERIFIED_TRUE
        && cam_set_[frm_idx].mask.at<uchar>(y, x) == my::VERIFIED_TRUE) {
      Eigen::Vector3d tmp_hw, tmp_h1w, tmp_hw1, tmp_h1w1;
      Eigen::Vector3d obj_hw, obj_h1w, obj_hw1, obj_h1w1;
      tmp_hw << (x - dx) / fx, (y - dy) / fy, 1.0;
      tmp_h1w << (x - dx) / fx, (y + 1 - dy) / fy, 1.0;
      tmp_hw1 << (x + 1 - dx) / fx, (y - dy) / fy, 1.0;
      tmp_h1w1 << (x + 1 - dx) / fx, (y + 1 - dy) / fy, 1.0;
      obj_hw = gauss_depth.at<double>(y, x) * tmp_hw;
      obj_h1w = gauss_depth.at<double>(y + 1, x) * tmp_h1w;
      obj_hw1 = gauss_depth.at<double>(y, x + 1) * tmp_hw1;
      obj_h1w1 = gauss_depth.at<double>(y + 1, x + 1) * tmp_h1w1;
      Eigen::Vector3d vec_ul2dr, vec_dl2ur;
      vec_ul2dr = obj_h1w1 - obj_hw;
      vec_dl2ur = obj_hw1 - obj_h1w;
      Eigen::Vector3d norm_vec = vec_ul2dr.cross(vec_dl2ur);
      norm_vec = norm_vec / norm_vec.norm();

//      vertex_set_[frm_idx].vertex_val_(i, 0)
//          = cam_set_[frm_idx].depth.at<double>(y, x);
      double norm_weight = norm_vec.transpose() * calib_set_.light_vec_;
      if (norm_weight < 0)
        norm_weight = - norm_weight;
      vertex_set_[frm_idx].vertex_val_(i, 0) = norm_weight;

      vertex_set_[frm_idx].vertex_val_.block(i, 1, 1, 3) = norm_vec.transpose();
      vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
    }
  }
  LOG(INFO) << "End: SetFirstFrameVertex(" << frm_idx << ");";
}

// Set Mask mat according to pixel from I_obs
void Reconstructor::SetMaskMatFromIobs(int frm_idx) {
  LOG(INFO) << "Start: SetMaskMatFromIobs(" << frm_idx << ");";
  cam_set_[frm_idx].mask.create(kCamHeight, kCamWidth, CV_8UC1);
  // Set mask initial value by intensity:
  //   my::INITIAL_FALSE for dark part
  //   my::INITIAL_TRUE for light part
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].img_obs.at<uchar>(h, w) <= kMaskIntensityThred) {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::INITIAL_FALSE;
      } else {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::INITIAL_TRUE;
      }
    }
  }
  // FloodFill every point
  //   For my::INITIAL_FALSE:
  //     area < Thred: Should be set to my::VERIFIED_TRUE
  //   for my::INITIAL_TRUE:
  //     area < Thred: Should be set to my::VERIFIED_FALSE
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      uchar val = cam_set_[frm_idx].mask.at<uchar>(h, w);
      if (val == my::INITIAL_FALSE) {
        int num = FloodFill(cam_set_[frm_idx].mask, h, w,
                            my::MARKED, my::INITIAL_FALSE);
        if (num <= kMaskMinAreaThred) {
          FloodFill(cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_TRUE, my::MARKED);
        } else {
          FloodFill(cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_FALSE, my::MARKED);
        }
      } else if (val == my::INITIAL_TRUE) {
        int num = FloodFill(cam_set_[frm_idx].mask, h, w,
                            my::MARKED, my::INITIAL_TRUE);
        if (num <= kMaskMinAreaThred) {
          FloodFill(cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_FALSE, my::MARKED);
        } else {
          FloodFill(cam_set_[frm_idx].mask, h, w,
                    my::VERIFIED_TRUE, my::MARKED);
        }
      }
    }
  }
  // Set result with hard_mask
  // cv::imshow("test", cam_set_[frm_idx].mask * 255);
  // cv::waitKey(0);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if ((hard_mask_.at<uchar>(h, w) == my::VERIFIED_FALSE)
          && (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_TRUE)) {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_FALSE;
      }
    }
  }
  LOG(INFO) << "End: SetMaskMatFromIobs(" << frm_idx << ");";
}

// Set vertex_set from last frm
void Reconstructor::PredictInitialShadeVertex(int frm_idx) {
  LOG(INFO) << "Start: PredictInitialShadeVertex(" << frm_idx << ");";
  // v_i(t) = v_i(t-1), for most has history vertex
  for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
    int x = vertex_set_[frm_idx].pos_(i, 0);
    int y = vertex_set_[frm_idx].pos_(i, 1);
    if (cam_set_[frm_idx].mask.at<uchar>(y, x) != my::VERIFIED_TRUE) {
      vertex_set_[frm_idx].vertex_val_(i, 0) = -1;
      vertex_set_[frm_idx].valid_(i, 0) = my::VERIFIED_FALSE;
      continue;
    }

    // If last frame has vertex_value, v_i(t) = v_i(t-1)
    if (vertex_set_[frm_idx - 1].valid_(i) == my::VERIFIED_TRUE) {
      vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
      vertex_set_[frm_idx].vertex_val_.block(i, 0, 1, 4)
          = vertex_set_[frm_idx - 1].vertex_val_.block(i, 0, 1, 4);
    } else {
      vertex_set_[frm_idx].valid_(i) = my::INITIAL_TRUE;
    }
  }
  // Find nearest vertex
  for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
    int x = vertex_set_[frm_idx].pos_(i, 0);
    int y = vertex_set_[frm_idx].pos_(i, 1);
    if (vertex_set_[frm_idx].valid_(i) != my::INITIAL_TRUE) {
      continue;
    }
    Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
        = vertex_set_[frm_idx].FindkNearestVertex(x, y, 2);
    int idx_nbr = (int)vertex_nbr(1, 0);
    vertex_set_[frm_idx].vertex_val_.block(i, 0, 1, 4)
        = vertex_set_[frm_idx].vertex_val_.block(idx_nbr, 0, 1, 4);
  }

  for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
    int x = vertex_set_[frm_idx].pos_(i, 0);
    int y = vertex_set_[frm_idx].pos_(i, 1);
    if (vertex_set_[frm_idx].valid_(i) == my::INITIAL_TRUE) {
      vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
    }
  }
  LOG(INFO) << "End: PredictInitialShadeVertex(" << frm_idx << ");";
}

// Fill shade_mat from vertex_set
void Reconstructor::FillShadeMatFromVertex(int frm_idx) {
  LOG(INFO) << "Start: FillShadeMatFromVertex(" << frm_idx << ");";
  cam_set_[frm_idx].shade_mat.create(kCamHeight, kCamWidth, CV_64FC1);
  cam_set_[frm_idx].shade_mat.setTo(0);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; ++w) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
          = vertex_set_[frm_idx].FindkNearestVertex(w, h, kNearestPoints + 1);
      Eigen::Matrix<double, Eigen::Dynamic, 1> weight
          = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(kNearestPoints, 1);
      double sum_val = 0;
      for (int i = 0; i < kNearestPoints; i++) {
        weight(i) = pow((1 - vertex_nbr(i, 1) / vertex_nbr(kNearestPoints, 1)), 2);
        sum_val += weight(i);
      }
      weight = weight / sum_val;
      double shade_val = 0;
      for (int i = 0; i < kNearestPoints; i++) {
        int idx_nbr = vertex_nbr(i, 0);
//        double vertex_val = vertex_set_[frm_idx].vertex_val_(idx_nbr, 0);
//        double weight_t = weight(i);
        shade_val += weight(i) * vertex_set_[frm_idx].vertex_val_(idx_nbr, 0);
      }
      cam_set_[frm_idx].shade_mat.at<double>(h, w) = shade_val;
    }
  }
//  ShowMat(&cam_set_[frm_idx].shade_mat, "shade_mat", 0, 0.0, 1.0);
  LOG(INFO) << "End: FillShadeMatFromVertex(" << frm_idx << ");";
}

// Classify intensity class. Use K-means on each blocks.
void Reconstructor::RecoIntensityClass(int frm_idx) {
  LOG(INFO) << "Start: RecoIntensityClass(" << frm_idx << ")";

  cam_set_[frm_idx].km_center
      = Eigen::Matrix<double, kIntensityClassNum, Eigen::Dynamic>::Zero(
      kIntensityClassNum, kKMBlockHeightNum*kKMBlockWidthNum);
  cam_set_[frm_idx].img_class.create(kCamHeight, kCamWidth, CV_8UC1);
  cam_set_[frm_idx].img_class.setTo(0);
  cam_set_[frm_idx].img_class_p.create(kCamHeight, kCamWidth, CV_64FC1);
  cam_set_[frm_idx].img_class_p.setTo(0);

  cv::Mat tmp_mark(kCamHeight, kCamWidth, CV_8UC1);
  for (int h_i = 0; h_i < kKMBlockHeightNum; h_i++) {
    for (int w_i = 0; w_i < kKMBlockWidthNum; w_i++) {
      // Set range of k-means
      int h_start = kKMBlockHeight*(h_i - 1);
      int w_start = kKMBlockWidth*(w_i - 1);
      int h_end = kKMBlockHeight*(h_i + 2);
      int w_end = kKMBlockWidth*(w_i + 2);
      h_start = h_start >= 0 ? h_start : 0;
      w_start = w_start >= 0 ? w_start : 0;
      h_end = h_end <= kCamHeight ? h_end : kCamHeight;
      w_end = w_end <= kCamWidth ? w_end : kCamWidth;

      // Set initial value of k_means center
      Eigen::Matrix<double, kIntensityClassNum, 1> k_center
          = Eigen::Matrix<double, kIntensityClassNum, 1>::Zero();
      Eigen::Matrix<double, kIntensityClassNum, 1> k_new_center
          = Eigen::Matrix<double, kIntensityClassNum, 1>::Zero();
      int idx_i = h_i * kKMBlockWidthNum + w_i;
      if (frm_idx == 0) {
        for (int c = 0; c < kIntensityClassNum; c++) {
          k_center(c, 0) = 255 * ((double)c / (double)(kIntensityClassNum - 1));
        }
      } else {
        k_center = cam_set_[frm_idx - 1].km_center.block<kIntensityClassNum, 1>(
            0, idx_i);
      }

      // Search
      double sum_thred = 2.0; // center diff
      bool flag = true;
      while (flag) {
        // Set sum_val & sum_num
        Eigen::Matrix<double, kIntensityClassNum, 2> k_info
            = Eigen::Matrix<double, kIntensityClassNum, 2>::Zero();
        // 1. For every pixel, find the intensity class.
        for (int h = h_start; h < h_end; h++) {
          for (int w = w_start; w < w_end; w++) {
            // Find class
            if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE) {
              continue;
            }
            double min_val = 255.0;
            int min_idx = -1;
            double img_obs_k = (double)cam_set_[frm_idx].img_obs.at<uchar>(h, w);
            double norm_weight = cam_set_[frm_idx].shade_mat.at<double>(h, w);
            double img_pat = img_obs_k / norm_weight;
            img_pat = img_pat <= 255.0 ? img_pat : 255.0;
            img_pat = img_pat >= 0 ? img_pat : 0.0;
            for (int c = 0; c < kIntensityClassNum; c++) {
              double diff_val = std::abs(k_center(c, 0) - img_pat);
              min_val = diff_val <= min_val ? diff_val : min_val;
              min_idx = diff_val <= min_val ? c : min_idx;
            }
            if (min_idx < 0) {
              LOG(ERROR) << "Invalid min_idx";
            }
            tmp_mark.at<uchar>(h, w) = (uchar)min_idx;
            k_info(min_idx, 0) = k_info(min_idx, 0) + img_pat;
            k_info(min_idx, 1) = k_info(min_idx, 1) + 1;
          }
        }
        // 2. Calculate new center
        double sum_diff = 0;
        for (int c = 0; c < kIntensityClassNum; c++) {
          if (k_info(c, 1) == 0) {
            k_new_center(c, 0) = k_center(c, 0);
          } else {
            k_new_center(c, 0) = k_info(c, 0) / k_info(c, 1);
            sum_diff += std::abs(k_new_center(c, 0) - k_center(c, 0));
          }
        }
        if (sum_diff <= sum_thred) {
          flag = false;
        }
        k_center = k_new_center;
      }

      // Finish & Fill
      cam_set_[frm_idx].km_center.block(0, idx_i, kIntensityClassNum, 1)
          = k_center;
      for (int h = h_i*kKMBlockHeight; h < (h_i+1)*kKMBlockHeight; h++) {
        for (int w = w_i*kKMBlockWidth; w < (w_i+1)*kKMBlockWidth; w++) {
          if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
            continue;
          // Calculate img_pat
          double img_obs_k = (double)cam_set_[frm_idx].img_obs.at<uchar>(h, w);
          double norm_weight = cam_set_[frm_idx].shade_mat.at<double>(h, w);
          double img_pat = img_obs_k / norm_weight;
          img_pat = img_pat <= 255.0 ? img_pat : 255.0;
          img_pat = img_pat >= 0 ? img_pat : 0.0;
          // set class
          int c = tmp_mark.at<uchar>(h, w);
          cam_set_[frm_idx].img_class.at<uchar>(h, w) = (uchar)c;
          // set probability
          double center = cam_set_[frm_idx].km_center(c, idx_i);
          double p = 0.0;
          if (c == 0 && img_pat <= center) {
            p = 1.0;
          } else if (c == kIntensityClassNum - 1 && img_pat >= center) {
            p = 1.0;
          } else {
            double nbr_center = (img_pat < center)
                                ? cam_set_[frm_idx].km_center(c - 1, idx_i)
                                : cam_set_[frm_idx].km_center(c + 1, idx_i);
            double divide = (center + nbr_center) / 2;
            p = 1 - std::abs(center - img_pat) / std::abs(center - divide);
          }
          cam_set_[frm_idx].img_class_p.at<double>(h, w) = p;
        }
      }
    }
  }

  // Fix img_class to discard unstable points
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
//        continue;
//      int h_i = h / kKMBlockHeight;
//      int w_i = w / kKMBlockWidth;
//      int idx_i = h_i * kKMBlockWidthNum + w_i;
//      double img_obs_k = (double)cam_set_[frm_idx].img_obs.at<uchar>(h, w);
//      double norm_weight = cam_set_[frm_idx].shade_mat.at<double>(h, w);
//      double img_pat = img_obs_k / norm_weight;
//      img_pat = img_pat <= 255.0 ? img_pat : 255.0;
//      img_pat = img_pat >= 0 ? img_pat : 0.0;
//      int c = cam_set_[frm_idx].img_class.at<uchar>(h, w);
//      double center = cam_set_[frm_idx].km_center(c, idx_i);
//      // TODO:Change this into mutli-color version
//      double thred = std::abs(cam_set_[frm_idx].km_center(1, idx_i)
//                              - cam_set_[frm_idx].km_center(0, idx_i)) / 4;
////      if (std::abs(center - img_pat) > thred) {
////        cam_set_[frm_idx].img_class.at<uchar>(h, w) = kIntensityClassNum;
////      }
//      if (c == 0 && img_pat >= center) {
//        cam_set_[frm_idx].img_class.at<uchar>(h, w) = kIntensityClassNum;
//      } else if (c == 1 && img_pat <= center) {
//        cam_set_[frm_idx].img_class.at<uchar>(h, w) = kIntensityClassNum;
//      }
//    }
//  }
  LOG(INFO) << "End: RecoIntensityClass(" << frm_idx << ")";
}

// Create nearby IntensitySlots.
// Input: hard_mask, pattern, epi_para
// Output: IntensitySlots
void Reconstructor::GenerateIntensitySlots(int frm_idx) {
  LOG(INFO) << "Start: GenerateIntensitySlots(" << frm_idx << ")";
  if (frm_idx < 0 || frm_idx >= kFrameNum) {
    LOG(ERROR) << "Invalid frm_idx: " << frm_idx;
    return;
  }
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE) {
        continue;
      }
      if (cam_slots_->slots_[h][w] != nullptr) {
        continue;
      }

      // Initial depth
      double xpro_start = GetXproFromDepth(kDepthMin, h, w, &calib_set_);
      double xpro_end = GetXproFromDepth(kDepthMax, h, w, &calib_set_);
      double xpro_left = xpro_end;
      double xpro_right = xpro_start;

      // Create slots for pixel
      int last_c = -1;
      cam_slots_->slots_[h][w] = new IntensitySlot[kIntensityClassNum + 1];
      DepthSegment tmp_seg[kIntensityClassNum + 1];
      for (int x = (int)xpro_start; x <= (int) (xpro_end + 1); x++) {
        // Make sure x,y is legal
        if (x < 0 || x >= kProWidth)
          continue;
        int y = (int) GetYproFromXpro(x, h, w, epi_A_mat_, epi_B_mat_);
        if (y < 0 || y >= kProHeight)
          continue;
        // Update xpro_left & xpro_right
        xpro_left = (xpro_left < x) ? xpro_left : x;
        xpro_right = (xpro_right > x) ? xpro_right : x;
        // Find color c at pro_pixel(y, x)
        double pat_val = pattern_.at<double>(y, x);
        int now_c = pat_val >= inten_thred_ ? 1 : 0;
        // Check:
        if (last_c < 0) {
          double divide_depth = GetDepthFromXpro(x, h, w, &calib_set_);
          tmp_seg[now_c].start = divide_depth;
          last_c = now_c;
        } else if (last_c != now_c) {
          double divide_depth = GetDepthFromXpro(x - 0.5, h, w, &calib_set_);
          tmp_seg[last_c].end = divide_depth;
          cam_slots_->slots_[h][w][last_c].InsertSegment(tmp_seg[last_c]);

          tmp_seg[last_c].Clear();
          tmp_seg[now_c].start = divide_depth;
          last_c = now_c;
        }
      }
      if (last_c < 0 || xpro_left >= xpro_right) {
        LOG(FATAL) << "No valid pattern found: (" << h << "," << w << ")";
      } else {
        double divide_depth = GetDepthFromXpro(xpro_right, h, w, &calib_set_);
        tmp_seg[last_c].end = divide_depth;
        cam_slots_->slots_[h][w][last_c].InsertSegment(tmp_seg[last_c]);
      }

      // Add to unconstraint slots
      double depth_start = GetDepthFromXpro(xpro_left, h, w, &calib_set_);
      double depth_end = GetDepthFromXpro(xpro_right, h, w, &calib_set_);
      tmp_seg[kIntensityClassNum].start = depth_start;
      tmp_seg[kIntensityClassNum].end = depth_end;
      cam_slots_->slots_[h][w][kIntensityClassNum].InsertSegment(
          tmp_seg[kIntensityClassNum]);
    }
  }
  LOG(INFO) << "End: GenerateIntensitySlots(" << frm_idx << ")";
}

// Fill initial depth val for every pixel (Without intensity constraint)
void Reconstructor::PredictInitialDepthVal(int frm_idx) {
  LOG(INFO) << "Start: PredictInitialDepthVal(" << frm_idx << ")";
  if ((frm_idx <= 0) || frm_idx >= kFrameNum) {
    LOG(ERROR) << "Invalid frm_idx: " << frm_idx;
    return;
  }
  cam_set_[frm_idx].depth.create(kCamHeight, kCamWidth, CV_64FC1);
  double error_thred = 6.0;
  int check_rad = 5;
  for (int h = 0; h < kCamHeight; ++h) {
    for (int w = 0; w < kCamWidth; ++w) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE) {
        cam_set_[frm_idx].depth.at<double>(h, w) = -1;
        continue;
      }
      // 1. d_k(t-1) has value, d_k(t) = d_k(t-1)
      if (cam_set_[frm_idx-1].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
        cam_set_[frm_idx].depth.at<double>(h, w)
            = cam_set_[frm_idx-1].depth.at<double>(h, w);
        continue;
      }
      // 2. d_k(t-1) does not have value. Find nearest point in d(t-1).
      int rad = 1;
      bool break_flag = false;
      double depth_val = -1;
      //double norm_weight = -1;
      int h_s = h - rad;
      int w_s = w - rad;
      while (!break_flag) {
        // Check
        if (h_s >= 0 && h_s < kCamHeight && w_s >= 0 && w_s < kCamWidth) {
          if (cam_set_[frm_idx - 1].mask.at<uchar>(h_s, w_s) == my::VERIFIED_TRUE) {
            depth_val = cam_set_[frm_idx - 1].depth.at<double>(h_s, w_s);
            //norm_weight = cam_set_[frm_idx - 1].shade_mat.at<double>(h_s, w_s);
            break_flag = true;
            continue;
          }
        }
        // Next
        if (h_s == h - rad && w_s < w + rad) {
          w_s++;
        } else if (w_s == w + rad && h_s < h + rad) {
          h_s++;
        } else if (h_s == h + rad && w_s > w - rad) {
          w_s--;
        } else if (w_s == w - rad && h_s > h - rad + 1) {
          h_s--;
        } else if (h_s == h - rad + 1 && w_s == w - rad) {
          rad++;
          h_s = h - rad;
          w_s = w - rad;
        } else {
          LOG(ERROR) << "Logic problem.";
        }
      }
      cam_set_[frm_idx].depth.at<double>(h, w) = depth_val;
    }
  }
//  ShowMat(&cam_set_[frm_idx].depth, "Depth_Init", 0, 25.0, 35.0);
  LOG(INFO) << "End: PredictInitialDepthVal(" << frm_idx << ")";
}

// Set depth val with intensity constraint
void Reconstructor::RefineInitialDepthVal(int frm_idx) {
  LOG(INFO) << "Start: RefineInitialDepthVal(" << frm_idx << ")";
  cam_set_[frm_idx].pointer = ImgMatrix::Zero(kCamHeight, kCamWidth);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      double depth = cam_set_[frm_idx].depth.at<double>(h, w);
      int c = cam_set_[frm_idx].img_class.at<uchar>(h, w);

      cam_set_[frm_idx].pointer(h, w)
          = cam_slots_->slots_[h][w][c].GetNearestPointerFromDepth(depth);

      if (cam_set_[frm_idx].pointer(h, w) < 0) {
        cam_set_[frm_idx].img_class.at<uchar>(h, w) = kIntensityClassNum;
        c = cam_set_[frm_idx].img_class.at<uchar>(h, w);
        cam_set_[frm_idx].pointer(h, w)
            = cam_slots_->slots_[h][w][kIntensityClassNum].GetNearestPointerFromDepth(depth);
        if (cam_set_[frm_idx].pointer(h, w) < 0)
          LOG(ERROR) << "Empty slot at: " << h << "," << w;
      }

      double new_depth = cam_slots_->slots_[h][w][c].GetDepthFromPointer(
          cam_set_[frm_idx].pointer(h, w));

      cam_set_[frm_idx].depth.at<double>(h, w) = new_depth;
    }
  }

  LOG(INFO) << "End: RefineInitialDepthVal(" << frm_idx << ")";
}

// Optimize Depth mat by spatial & temporal info
void Reconstructor::OptimizeDepthNode(int frm_idx) {
  LOG(INFO) << "Start: OptimizeDepthNode(" << frm_idx << ")";

  // parameters:
  double omega_s = 15.0;
  double omega_t = 3.0;
  ceres::Problem problem;
  double weight_t = 0.1;

  // For history:
  int t_start = (frm_idx - kTemporalWindowSize >= 0) ? frm_idx - kTemporalWindowSize : 0;
  int t_now = frm_idx;
  int block_num = 0;
  for (int t_cen = t_start; t_cen <= t_now; t_cen++) {
    for (int idx_cen = 0; idx_cen < node_set_[t_cen].len_; idx_cen++) {
      if (node_set_[t_cen].valid_(idx_cen, 0) != my::VERIFIED_TRUE)
        continue;
      int h_cen, w_cen;
      node_set_[t_cen].GetNodeCoordByIdx(idx_cen, &h_cen, &w_cen);
      int pos_x = node_set_[t_cen].pos_(idx_cen, 0);
      int pos_y = node_set_[t_cen].pos_(idx_cen, 1);
      uchar c_cen = cam_set_[t_cen].img_class.at<uchar>(pos_y, pos_x);
      for (int h_s = h_cen - 1; h_s <= h_cen + 1; h_s++) {
        for (int w_s = w_cen - 1; w_s <= w_cen + 1; w_s++) {
          for (int t_s = t_start; t_s <= t_now; t_s++) {
            if (h_s < 0 || h_s >= node_set_[t_s].block_height_
                || w_s < 0 || w_s >= node_set_[t_s].block_width_)
              continue;
            int idx_s = h_s * node_set_[t_s].block_width_ + w_s;
            if (idx_s == idx_cen && t_s == t_cen)
              continue;
            if (node_set_[t_s].valid_(idx_s, 0) != my::VERIFIED_TRUE)
              continue;

            // Calculate dist
            int pos_s_x = node_set_[t_s].pos_(idx_s, 0);
            int pos_s_y = node_set_[t_s].pos_(idx_s, 1);
            uchar c_s = cam_set_[t_s].img_class.at<uchar>(pos_s_y, pos_s_x);
            double dist_s = sqrt(pow(pos_x - pos_s_x, 2) + pow(pos_y - pos_s_y, 2));
            double dist_t = sqrt(pow(t_s - t_cen, 2));

            // Add block
            DepthRegConstraint::DepthRegCostFunction * cost_function
                = DepthRegConstraint::Create(
                    dist_s, dist_t, omega_s, omega_t,
                    &(cam_slots_->slots_[pos_y][pos_x][c_cen]),
                    &(cam_slots_->slots_[pos_s_y][pos_s_x][c_s]),
                    Num2Str(block_num));
            problem.AddResidualBlock(cost_function, nullptr,
                                     &(node_set_[t_cen].val_.data()[idx_cen]),
                                     &(node_set_[t_s].val_.data()[idx_s]));
            IntensitySlot * p_a = &(cam_slots_->slots_[pos_y][pos_x][c_cen]);
            IntensitySlot * p_b = &(cam_slots_->slots_[pos_s_y][pos_s_x][c_s]);
            if (p_a == nullptr) {
              LOG(ERROR) << "Empty pointer a";
            }
            if (p_b == nullptr) {
              LOG(ERROR) << "Empty pointer b";
            }
            block_num++;
          }
        }
      }
    }
  }

  // Solve
  ceres::Solver::Options options;
  options.gradient_tolerance = 1e-10;
  options.function_tolerance = 1e-3;
//  options.min_relative_decrease = 1e1;
  options.parameter_tolerance = 1e-14;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;

  LOG(INFO) << "Stat: OptimizeDepthNode(" << frm_idx << "), Begin opt with block_num = " << block_num;
  ceres::Solve(options, &problem, &summary);

  LOG(INFO) << summary.BriefReport();
  std::cout << summary.BriefReport() << std::endl;
  std::cout << summary.message << std::endl;
  LOG(INFO) << "End: OptimizeDepthNode(" << frm_idx << ")";

  // Fill depth_mat
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
//        continue;
//      int c = cam_set_[frm_idx].img_class.at<uchar>(h, w);
//      cam_set_[frm_idx].depth.at<double>(h, w)
//          = cam_slots_->slots_[h][w][c].GetDepthFromPointer(
//          cam_set_[frm_idx].pointer(h, w));
//    }
//  }
}

// Set node[t] from pointer[t]
void Reconstructor::SetNodeFromDepthVal(int frm_idx) {
  for (int i = 0; i < node_set_[frm_idx].len_; i++) {
    int h_ord, w_ord;
    node_set_[frm_idx].GetNodeCoordByIdx(i, &h_ord, &w_ord);
    int h = h_ord * kNodeBlockSize;
    int w = w_ord * kNodeBlockSize;

    // Set pos & val: find maxP pixel
    int pos_x = -1, pos_y = -1;
    double max_p = 0.0;
    double min_dis = 1000.0;
    double center_x = w + ((double)kNodeBlockSize - 1.0) / 2;
    double center_y = h + ((double)kNodeBlockSize - 1.0) / 2;
    for (int y = h; y < h + kNodeBlockSize; y++) {
      for (int x = w; x < w + kNodeBlockSize; x++) {
        if (cam_set_[frm_idx].mask.at<uchar>(y, x) != my::VERIFIED_TRUE) {
          continue;
        }
        double p = cam_set_[frm_idx].img_class_p.at<double>(y, x);
        double dis = pow(x - center_x, 2) + pow(y - center_y, 2);
        if ((p > max_p) || (p == max_p && dis < min_dis)) {
          max_p = p;
          pos_x = x; pos_y = y;
          min_dis = dis;
        }
      }
    }
    if (pos_x < 0 || pos_y < 0) {
      node_set_[frm_idx].valid_(i, 0) = my::VERIFIED_FALSE;
      continue;
    }

    node_set_[frm_idx].SetNodePos(i, pos_x, pos_y);
    double pointer_val = cam_set_[frm_idx].pointer(pos_y, pos_x);
    double depth_val = cam_set_[frm_idx].depth.at<double>(pos_y, pos_x);
    uchar c = cam_set_[frm_idx].img_class.at<uchar>(pos_y, pos_x);
    double refix_depth = cam_slots_->slots_[pos_y][pos_x][c].GetDepthFromPointer(pointer_val);
    node_set_[frm_idx].val_(i, 0) = pointer_val;
    node_set_[frm_idx].valid_(i, 0) = my::VERIFIED_TRUE;
  }
}

// Set depth[t] & pointer[t] from node[t], class[t]
void Reconstructor::SetDepthValFromNode(int frm_idx) {
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;

      // Find nearest nbr
      Eigen::Matrix<double, Eigen::Dynamic, 2> nbr_set
          = node_set_[frm_idx].FindkNearestNodes(w, h, kNearestPoints + 1);
      // Calculate weight
      Eigen::Matrix<double, Eigen::Dynamic, 1> nbr_weight
          = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(kNearestPoints, 1);
      CalculateNbrWeight(nbr_set, &nbr_weight, kNearestPoints);
      // Sum as depth
      double depth_sum = 0;
      for (int k = 0; k < kNearestPoints; k++) {
        int idx = nbr_set(k, 0);
        double p_val = node_set_[frm_idx].val_(idx, 0);
        int x = node_set_[frm_idx].pos_(idx, 0);
        int y = node_set_[frm_idx].pos_(idx, 1);
        uchar c = cam_set_[frm_idx].img_class.at<uchar>(y, x);
        double depth
            = cam_slots_->slots_[y][x][c].GetDepthFromPointer(p_val);
        depth_sum += depth * nbr_weight(k, 0);
      }
      cam_set_[frm_idx].depth.at<double>(h, w) = depth_sum;
    }
  }
}

// Generate I_est by pattern
void Reconstructor::GenerateIestFromDepth(int frm_idx) {
  LOG(INFO) << "Start: GenerateIestFromDepth(" << frm_idx << ")";
  cam_set_[frm_idx].img_est.create(kCamHeight, kCamWidth, CV_8UC1);
  cam_set_[frm_idx].img_est.setTo(0);

  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE) {
        continue;
      }
      // Get depth
      double depth = cam_set_[frm_idx].depth.at<double>(h, w);
      if (depth <= 0)
        continue;
      double x_pro = GetXproFromDepth(depth, h, w, &calib_set_);
      double y_pro = GetYproFromXpro(x_pro, h, w, epi_A_mat_, epi_B_mat_);
      // Get img_intensity
      double img_k_head;
      pat_grid_->Evaluate(y_pro, x_pro, &img_k_head);
      double norm_weight = cam_set_[frm_idx].shade_mat.at<double>(h, w);
      cam_set_[frm_idx].img_est.at<uchar>(h, w) = uchar(img_k_head * norm_weight);
    }
  }

//  if (frm_idx > 0) {
//    std::cout << "Debug:" << std::endl;
//    int h_p = 653 - 1, w_p = 627 - 1;
//    std::cout << "pointer[" << h_p << "][" << w_p << "]=" << cam_set_[frm_idx].pointer(h_p, w_p);
//  }

//  ShowMat<uchar>(&cam_set_[frm_idx].img_est, "img_est", 0, 0, 255);
  LOG(INFO) << "End: GenerateIestFromDepth(" << frm_idx << ")";
}

// Optimize Shading part by current frame & I_est
void Reconstructor::OptimizeShadingMat(int frm_idx) {
  LOG(INFO) << "Start: OptimizeShadingMat(" << frm_idx << ")";
  double weight_t = 1.0;
  ceres::Problem shade_problem;

  // Add blocks
  int data_blocks = 0;
  double* data_set = vertex_set_[frm_idx].vertex_val_.data();
  for (int h = 0; h < kCamHeight; ++h) {
    for (int w = 0; w < kCamWidth; ++w) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      // Add Data Term
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
          = vertex_set_[frm_idx].FindkNearestVertex(w, h, kNearestPoints + 1);
      double img_obs_k = (double)cam_set_[frm_idx].img_obs.at<uchar>(h, w);
      double img_est_k = (double)cam_set_[frm_idx].img_est.at<uchar>(h, w);
      double shade_k = (double)cam_set_[frm_idx].shade_mat.at<double>(h, w);
      double img_pattern_k = img_est_k / shade_k;
      std::vector<double*> parameter_blocks;

      ShadeDataConstraint::ShadeDataCostFunction* cost_function
          = ShadeDataConstraint::Create(
              img_obs_k, img_pattern_k, kNearestPoints, vertex_nbr, data_set,
              &parameter_blocks);
      if (cost_function != nullptr) {
        shade_problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);
        data_blocks++;
      } else {
        LOG(ERROR) << "Invalid cost function. (" << h << "," << w << ")";
      }
    }
  }

  // optimize
  ceres::Solver::Options shade_options;
  shade_options.gradient_tolerance = 1e-10;
  shade_options.function_tolerance = 1e-10;
  shade_options.min_relative_decrease = 1e-5;
  shade_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  shade_options.max_num_iterations = 100;
  shade_options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary shade_summary;
  LOG(INFO) << "Stat: OptimizeShadingMat(" << frm_idx << "), Begin opt.";
  ceres::Solve(shade_options, &shade_problem, &shade_summary);

  LOG(INFO) << shade_summary.BriefReport();
  std::cout << shade_summary.BriefReport() << std::endl;
  LOG(INFO) << "End: OptimizeShadingMat(" << frm_idx << ")";
}

// Output all the result for further analysis
void Reconstructor::OutputResult(int frm_idx) {
  LOG(INFO) << "Start: OutputResult(" << frm_idx << ")";
  bool status = true;
  // Depth mat
  std::string depth_txt_name
      = output_file_path_ + depth_file_path_ + depth_file_name_
        + Num2Str(frm_idx) + ".txt";
  status = SaveMatToTxt(depth_txt_name, cam_set_[frm_idx].depth);
  if (status) LOG(INFO) << depth_txt_name << " success.";

  // Shade mat
  std::string shade_txt_name
      = output_file_path_ + "shade" + Num2Str(frm_idx) + ".txt";
  status = SaveMatToTxt(shade_txt_name, cam_set_[frm_idx].shade_mat);
  if (status) LOG(INFO) << shade_txt_name << " success.";
  // Shade mat show
  std::string shade_png_name
      = output_file_path_ + "shade" + Num2Str(frm_idx) + ".png";
  status = cv::imwrite(shade_png_name, cam_set_[frm_idx].shade_mat * 255.0);
  if (status) LOG(INFO) << shade_png_name << " success.";

  // img_obs
  std::string obs_img_name
      = output_file_path_ + "I_obs" + Num2Str(frm_idx) + ".png";
  status = cv::imwrite(obs_img_name, cam_set_[frm_idx].img_obs);
  if (status) LOG(INFO) << obs_img_name << " success.";
  // img_est
  std::string est_img_name
      = output_file_path_ + "I_est" + Num2Str(frm_idx) + ".png";
  status = cv::imwrite(est_img_name, cam_set_[frm_idx].img_est);
  if (status) LOG(INFO) << est_img_name << " success.";
  // pattern_class
  std::string class_img_name
      = output_file_path_ + "I_class" + Num2Str(frm_idx) + ".png";
  status = cv::imwrite(class_img_name, cam_set_[frm_idx].img_class);
  if (status) LOG(INFO) << class_img_name << " success.";

  // mask
  std::string mask_img_name
      = output_file_path_ + "mask" + Num2Str(frm_idx) + ".png";
  status = cv::imwrite(mask_img_name, cam_set_[frm_idx].mask);
  if (status) LOG(INFO) << class_img_name << " success.";

//  // pointer
//  std::string pointer_txt_name
//      = output_file_path_ + "pointer" + Num2Str(frm_idx) + ".txt";
//  if (frm_idx > 0) {
//    status = SaveImgMatToTxt(pointer_txt_name, cam_set_[frm_idx].pointer);
//  }

  // vertex_set
  std::string vertex_txt_name
      = output_file_path_ + "vertex_val" + Num2Str(frm_idx) + ".txt";
  status = SaveValToTxt(vertex_txt_name, vertex_set_[frm_idx].vertex_val_,
                        vertex_set_[frm_idx].block_height_,
                        vertex_set_[frm_idx].block_width_);
  if (status) LOG(INFO) << vertex_txt_name << " success.";
  // vertex_valid
  std::string valid_txt_name
      = output_file_path_ + "valid" + Num2Str(frm_idx) + ".txt";
  status = SaveVecUcharToTxt(valid_txt_name, vertex_set_[frm_idx].valid_,
                             vertex_set_[frm_idx].block_height_,
                             vertex_set_[frm_idx].block_width_);
  if (status) LOG(INFO) << valid_txt_name << " success.";

  // Node info
  std::string node_txt_name
      = output_file_path_ + "node" + Num2Str(frm_idx) + ".txt";
  status = node_set_[frm_idx].WriteToFile(node_txt_name);
  if (status) LOG(INFO) << node_txt_name << " success.";

  LOG(INFO) << "End: OutputResult(" << frm_idx << ")";
}

void Reconstructor::ReleaseSpace(int frm_idx) {
  LOG(INFO) << "Start: ReleaseSpace(" << frm_idx << ")";
  // Release CamMatSet:
  cam_set_[frm_idx].img_obs.release();
  cam_set_[frm_idx].img_class.release();
  cam_set_[frm_idx].img_class_p.release();
  cam_set_[frm_idx].shade_mat.release();
  cam_set_[frm_idx].img_est.release();
  cam_set_[frm_idx].x_pro.release();
  cam_set_[frm_idx].y_pro.release();
  cam_set_[frm_idx].depth.release();
  cam_set_[frm_idx].mask.release();
  cam_set_[frm_idx].pointer.resize(0, 0);
  cam_set_[frm_idx].km_center.resize(kIntensityClassNum, 0);
  cam_set_[frm_idx].norm_vec.resize(3, 0);
  // Release NodeSet
  node_set_[frm_idx].Clear();
  // Release vertex_set
  vertex_set_[frm_idx].Clear();
  LOG(INFO) << "End: ReleaseSpace(" << frm_idx << ")";
}

bool Reconstructor::Close() {
  return true;
}

/*
void Reconstructor::UpdateVertexFrame(int frm_idx) {

  if (frm_idx == 1) {
    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
      vertex_set_[frm_idx].frm_(i) = frm_idx;
    }
  }
  // For every vertex in mask set
  int s_rad = 7;
  double diff_thred = 10;
  double num_thred = 0.1;
  for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
    int x = vertex_set_[frm_idx].pos_(i, 0);
    int y = vertex_set_[frm_idx].pos_(i, 1);
    if (cam_set_[frm_idx].mask.at<uchar>(y, x) != my::VERIFIED_TRUE)
      continue;
    // Found history info
    int frm_old = vertex_set_[frm_idx - 1].frm_(i);
    if (cam_set_[frm_old].mask.at<uchar>(y, x) != my::VERIFIED_TRUE)
      continue;
    int valid_pix_sum = 0;
    int diff_pix_num = 0;
    for (int dh = -s_rad; dh <= s_rad; dh++) {
      for (int dw = -s_rad; dw <= s_rad; dw++) {
        int h = y + dh;
        int w = x + dw;
        if (h < 0 || h >= kCamHeight || w < 0 || w > kCamWidth)
          continue;
        if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE
            || cam_set_[frm_old].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
          continue;
        valid_pix_sum++;
        double I_obs_old = cam_set_[frm_old].img_obs.at<uchar>(h, w);
        double I_obs_now = cam_set_[frm_idx].img_obs.at<uchar>(h, w);
        if (std::abs(I_obs_old - I_obs_now) > diff_thred) {
          diff_pix_num++;
        }
      }
    }
    // Count diff num
    if ((double)diff_pix_num / (double)valid_pix_sum > num_thred) {
      vertex_set_[frm_idx].frm_(i) = frm_idx;
    } else {
      vertex_set_[frm_idx].frm_(i) = frm_old;
    }
  }
  // Save
  std::string file_name = output_file_path_ + vertex_file_path_
                          + "vertex_frm" + Num2Str(frm_idx) + ".txt";
  SaveFrmToTxt(
      file_name, vertex_set_[frm_idx].frm_,
      vertex_set_[frm_idx].block_height_, vertex_set_[frm_idx].block_width_);
}

void Reconstructor::NormalizeIobs(int frm_idx) {
  // Normalize obs image
  uchar max_val = 0;
  uchar min_val = 255;
//  cam_set_[frm_idx].img_obs.create(kCamHeight, kCamWidth, CV_8UC1);
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
//        if (cam_set_[frm_idx].img_obs_raw.at<uchar>(h, w) > max_val)
//          max_val = cam_set_[frm_idx].img_obs_raw.at<uchar>(h, w);
//        if (cam_set_[frm_idx].img_obs_raw.at<uchar>(h, w) < min_val)
//          min_val = cam_set_[frm_idx].img_obs_raw.at<uchar>(h, w);
//      } else {
//        cam_set_[frm_idx].img_obs.at<uchar>(h, w) = 0;
//      }
//    }
//  }
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
//        double ori_val = cam_set_[frm_idx].img_obs_raw.at<uchar>(h, w);
//        double new_val = (ori_val - min_val) / (double)(max_val - min_val);
//        cam_set_[frm_idx].img_obs.at<uchar>(h, w) = (uchar)(new_val * 255);
//      }
//    }
//  }
  cam_set_[frm_idx].img_obs_raw.copyTo(cam_set_[frm_idx].img_obs);
  cv::namedWindow("I_obs" + Num2Str(frm_idx));
  cv::imshow("I_obs" + Num2Str(frm_idx), cam_set_[frm_idx].img_obs);
  cv::imwrite(output_file_path_ + "I_obs" + Num2Str(frm_idx) + ".png", cam_set_[frm_idx].img_obs);
  cv::waitKey(2000);
  cv::destroyWindow("I_obs" + Num2Str(frm_idx));
}

void Reconstructor::SetVertexFromBefore(int frm_idx) {
  if (frm_idx == 0) { // Have depth & mask
    cv::Mat gauss_depth;
    cv::GaussianBlur(cam_set_[frm_idx].depth, gauss_depth, cv::Size(3, 3), 3.0);
    double fx = calib_set_.cam_mat(0, 0);
    double fy = calib_set_.cam_mat(1, 1);
    double dx = calib_set_.cam_mat(0, 2);
    double dy = calib_set_.cam_mat(1, 2);
    Eigen::Vector3d D = calib_set_.D;
    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
      int x = vertex_set_[frm_idx].pos_(i, 0);
      int y = vertex_set_[frm_idx].pos_(i, 1);
      if (cam_set_[frm_idx].mask.at<uchar>(y + 1, x + 1) == my::VERIFIED_TRUE
          && cam_set_[frm_idx].mask.at<uchar>(y + 1, x) == my::VERIFIED_TRUE
          && cam_set_[frm_idx].mask.at<uchar>(y, x + 1) == my::VERIFIED_TRUE
          && cam_set_[frm_idx].mask.at<uchar>(y, x) == my::VERIFIED_TRUE) {
        Eigen::Vector3d tmp_hw, tmp_h1w, tmp_hw1, tmp_h1w1;
        Eigen::Vector3d obj_hw, obj_h1w, obj_hw1, obj_h1w1;
        tmp_hw << (x - dx) / fx, (y - dy) / fy, 1.0;
        tmp_h1w << (x - dx) / fx, (y + 1 - dy) / fy, 1.0;
        tmp_hw1 << (x + 1 - dx) / fx, (y - dy) / fy, 1.0;
        tmp_h1w1 << (x + 1 - dx) / fx, (y + 1 - dy) / fy, 1.0;
        obj_hw = gauss_depth.at<double>(y, x) * tmp_hw;
        obj_h1w = gauss_depth.at<double>(y + 1, x) * tmp_h1w;
        obj_hw1 = gauss_depth.at<double>(y, x + 1) * tmp_hw1;
        obj_h1w1 = gauss_depth.at<double>(y + 1, x + 1) * tmp_h1w1;
        Eigen::Vector3d vec_ul2dr, vec_dl2ur;
        vec_ul2dr = obj_h1w1 - obj_hw;
        vec_dl2ur = obj_hw1 - obj_h1w;
        Eigen::Vector3d norm_vec = vec_ul2dr.cross(vec_dl2ur);
        norm_vec = norm_vec / norm_vec.norm();

        vertex_set_[frm_idx].vertex_val_(i, 0)
            = cam_set_[frm_idx].depth.at<double>(y, x);
        vertex_set_[frm_idx].vertex_val_.block(i, 1, 1, 3) = norm_vec.transpose();
        vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
      }
    }
  } else { // Not have
    double error_thred = 6.0;
    int check_rad = 5;
    // Check last frame: last frame have value, Set to VERIFIED_TRUE.
    // Otherwise, set to INITIAL_TRUE.
    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
      int x = vertex_set_[frm_idx].pos_(i, 0);
      int y = vertex_set_[frm_idx].pos_(i, 1);
      if (cam_set_[frm_idx].mask.at<uchar>(y, x) != my::VERIFIED_TRUE) {
        vertex_set_[frm_idx].vertex_val_(i, 0) = -1;
        vertex_set_[frm_idx].valid_(i, 0) = my::VERIFIED_FALSE;
        continue;
      }
//      if (frm_idx == 3 && x == 735 && y == 825) {
//        std::cout << "VERIFIED_TRUE." << std::endl;
//      }
      uchar set_flag = my::VERIFIED_TRUE;
      // 1. Last frame, vertex is not valid, set INITIAL
      if (vertex_set_[frm_idx - 1].valid_(i) != my::VERIFIED_TRUE)
        set_flag = my::INITIAL_TRUE;
      if (vertex_set_[frm_idx - 1].vertex_val_(i, 0) < 0)
        set_flag = my::INITIAL_TRUE;
      // 2. Last frame, mask is false, set to INITIAL
      if (cam_set_[frm_idx - 1].mask.at<uchar>(y, x) != my::VERIFIED_TRUE)
        set_flag = my::INITIAL_TRUE;
      // 3. All above passed, check I_obs & I_est
      if (set_flag == my::VERIFIED_TRUE) {
        double max_est = 0.0;
        double min_est = 255.0;
        double max_obs = 0.0;
        double min_obs = 255.0;
        int points_num = 0;
        for (int h = y - check_rad; h <= y + check_rad; h++) {
          for (int w = x - check_rad; w <= x + check_rad; w++) {
            if (cam_set_[frm_idx - 1].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
              points_num++;
              double i_obs_k = (double)cam_set_[frm_idx - 1].img_obs.at<uchar>(h, w);
              double i_est_k = (double)cam_set_[frm_idx - 1].img_est.at<uchar>(h, w);
              max_est = (max_est > i_est_k) ? max_est : i_est_k;
              min_est = (min_est < i_est_k) ? min_est : i_est_k;
              max_obs = (max_obs > i_obs_k) ? max_obs : i_obs_k;
              min_obs = (min_obs < i_obs_k) ? min_obs : i_obs_k;
            }
          }
        }
        double pat_diff = 0.0;
        double inten_diff = 0.0;
        for (int h = y - check_rad; h <= y + check_rad; h++) {
          for (int w = x - check_rad; w <= x + check_rad; w++) {
            if (cam_set_[frm_idx - 1].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
              double i_obs_k = (double)cam_set_[frm_idx - 1].img_obs.at<uchar>(h, w);
              double i_est_k = (double)cam_set_[frm_idx - 1].img_est.at<uchar>(h, w);
              inten_diff += std::abs(i_obs_k - i_est_k);
              double i_obs_k_n = (i_obs_k - min_obs) / (max_obs - min_obs);
              double i_est_k_n = (i_est_k - min_est) / (max_est - min_est);
              pat_diff += std::abs(i_obs_k_n - i_est_k_n);
            }
          }
        }
        pat_diff /= points_num;
        inten_diff /= points_num;
//          std::cout << pat_diff << " " << inten_diff << std::endl;
        if (inten_diff*pat_diff > error_thred) {
          set_flag = my::INITIAL_TRUE;
//            printf("Found.\n");
        }
      }
      vertex_set_[frm_idx].valid_(i) = set_flag;
      if (set_flag == my::VERIFIED_TRUE) {
        vertex_set_[frm_idx].vertex_val_.block(i, 0, 1, 4)
            = vertex_set_[frm_idx - 1].vertex_val_.block(i, 0, 1, 4);
      }
    }

    // Show
    cv::Mat show_mat;
    cam_set_[frm_idx - 1].img_est.copyTo(show_mat);
    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
      int y = vertex_set_[frm_idx].pos_(i, 1);
      int x = vertex_set_[frm_idx].pos_(i, 0);
      if (vertex_set_[frm_idx].valid_(i) == my::INITIAL_TRUE) {
//        std::cout << i << std::endl;
        cv::rectangle(show_mat, cv::Point(x-5, y-5), cv::Point(x+5, y+5), 255, 1);
      }
    }
    std::string valid_file_name = output_file_path_ + vertex_file_path_
                                   + "valid_status" + Num2Str(frm_idx) + ".png";
    cv::imwrite(valid_file_name, show_mat);

    // Fill valid mask: for every point with INITIAL_TRUE, find nearest point
    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
      int w = vertex_set_[frm_idx].pos_(i, 0);
      int h = vertex_set_[frm_idx].pos_(i, 1);
      if (vertex_set_[frm_idx].valid_(i) != my::INITIAL_TRUE) {
        continue;
      }
      // Depth: nearest depth in d^(t-1)
      int rad = 1;
      bool break_flag = false;
      double depth_val = -1;
      Eigen::Vector3d norm_vec = Eigen::Vector3d::Zero();
      while (!break_flag) {
        for (int r = -rad; r <= rad; r++) {
          int w_new = w + r;
          if (cam_set_[frm_idx - 1].mask.at<uchar>(h - rad, w_new)
              == my::VERIFIED_TRUE) {
            depth_val = cam_set_[frm_idx - 1].depth.at<double>(h - rad, w_new);
            norm_vec
                = cam_set_[frm_idx - 1].norm_vec.block<3, 1>(
                0, (h-rad)*kCamWidth + w_new);
            break_flag = true;
            break;
          }
          if (cam_set_[frm_idx - 1].mask.at<uchar>(h + rad, w_new)
              == my::VERIFIED_TRUE) {
            depth_val = cam_set_[frm_idx - 1].depth.at<double>(h + rad, w_new);
            norm_vec
                = cam_set_[frm_idx - 1].norm_vec.block<3, 1>(
                0, (h+rad)*kCamWidth + w_new);
            break_flag = true;
            break;
          }
        }
        if (break_flag)
          break;
        for (int r = -rad+1; r <= rad-1; r++) {
          int h_new = h + r;
          if (cam_set_[frm_idx - 1].mask.at<uchar>(h_new, w - rad)
              == my::VERIFIED_TRUE) {
            depth_val = cam_set_[frm_idx - 1].depth.at<double>(h_new, w - rad);
            norm_vec
                = cam_set_[frm_idx - 1].norm_vec.block<3, 1>(
                0, h_new*kCamWidth + w - rad);
            break_flag = true;
            break;
          }
          if (cam_set_[frm_idx - 1].mask.at<uchar>(h_new, w + rad)
              == my::VERIFIED_TRUE) {
            depth_val = cam_set_[frm_idx - 1].depth.at<double>(h_new, w + rad);
            norm_vec
                = cam_set_[frm_idx - 1].norm_vec.block<3, 1>(
                0, h_new*kCamWidth + w + rad);
            vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
            break_flag = true;
            break;
          }
        }
        rad++;
      }
      // Norm: Find nearest vertex
//      Eigen::Matrix<double, Eigen::Dynamic, 2> nearest_ver
//          = vertex_set_[frm_idx].FindkNearestVertex(w, h, 1);
//      int nbr_idx = (int)nearest_ver(i, 0);
      // Set val
      vertex_set_[frm_idx].vertex_val_(i, 0) = depth_val;
      vertex_set_[frm_idx].vertex_val_.block(i, 1, 1, 3) = norm_vec.transpose();
      vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
    }
//    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) { // Check
//      if ((vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE)
//          && (vertex_set_[frm_idx].vertex_val_(i, 0) <= 0)) {
//        ErrorThrow("Vertex value less than 0. i=" + Num2Str(i));
//      }
//    }

//    // Spread: 4
//    uchar st1, st2, st3, st4;
//    st1 = vertex_set_[frm_idx].valid_(2418);
//    st2 = vertex_set_[frm_idx].valid_(2419);
//    st3 = vertex_set_[frm_idx].valid_(2503);
//    st4 = vertex_set_[frm_idx].valid_(2504);
//    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
//      if (vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE) {
//        int left_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_LEFT);
//        if ((left_idx > 0)
//            && (vertex_set_[frm_idx].valid_(left_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(left_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(left_idx) = my::NEIGHBOR_4;
//        }
//        int right_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_RIGHT);
//        if ((right_idx > 0)
//            && (vertex_set_[frm_idx].valid_(right_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(right_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(right_idx) = my::NEIGHBOR_4;
//        }
//        int down_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_DOWN);
//        if ((down_idx > 0)
//            && (vertex_set_[frm_idx].valid_(down_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(down_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(down_idx) = my::NEIGHBOR_4;
//        }
//        int up_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_UP);
//        if ((up_idx > 0)
//            && (vertex_set_[frm_idx].valid_(up_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(up_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(up_idx) = my::NEIGHBOR_4;
//        }
//      }
//    }
//
//    // Spread: 8
//    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
//      if (vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE) {
//        int ul_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_UP_LEFT);
//        if ((ul_idx > 0)
//            && (vertex_set_[frm_idx].valid_(ul_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(ul_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(ul_idx) = my::NEIGHBOR_8;
//        }
//        int ur_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_UP_RIGHT);
//        if ((ur_idx > 0)
//            && (vertex_set_[frm_idx].valid_(ur_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(ur_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(ur_idx) = my::NEIGHBOR_8;
//        }
//        int dr_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_DOWN_RIGHT);
//        if ((dr_idx > 0)
//            && (vertex_set_[frm_idx].valid_(dr_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(dr_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(dr_idx) = my::NEIGHBOR_8;
//        }
//        int dl_idx = vertex_set_[frm_idx].GetNeighborVertexIdxByIdx(
//            i, my::DIREC_DOWN_LEFT);
//        if ((dl_idx > 0)
//            && (vertex_set_[frm_idx].valid_(dl_idx) == my::VERIFIED_FALSE)) {
//          vertex_set_[frm_idx].vertex_val_(dl_idx)
//              = vertex_set_[frm_idx].vertex_val_(i);
//          vertex_set_[frm_idx].valid_(dl_idx) = my::NEIGHBOR_8;
//        }
//      }
//    }
//
//    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
//      if ((vertex_set_[frm_idx].valid_(i) == my::NEIGHBOR_4)
//          || (vertex_set_[frm_idx].valid_(i) == my::NEIGHBOR_8)) {
//        vertex_set_[frm_idx].valid_(i) = my::VERIFIED_TRUE;
//      }
//      if ((vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE)
//          && (vertex_set_[frm_idx].vertex_val_(i) <= 0)) {
//        ErrorThrow("Vertex value less than 0. i=" + Num2Str(i));
//      }
//    }
  }

  std::string vertex_file_name = output_file_path_ + vertex_file_path_
                                 + vertex_file_name_ + "_init" + Num2Str(frm_idx) + ".txt";
  SaveValToTxt(
      vertex_file_name, vertex_set_[frm_idx].vertex_val_,
      vertex_set_[frm_idx].block_height_, vertex_set_[frm_idx].block_width_);
}

bool Reconstructor::OptimizeVertexSet(int frm_idx) {
  bool status = true;
  ceres::Problem problem;
  double alpha_val = 1.0;
  double alpha_norm = 1.0;

  double* data_set = vertex_set_[frm_idx].vertex_val_.data();
  int* frm_num_set = vertex_set_[frm_idx].frm_.data();

  // Add blocks
  double fx = calib_set_.cam_mat(0, 0);
  double fy = calib_set_.cam_mat(1, 1);
  double dx = calib_set_.cam_mat(0, 2);
  double dy = calib_set_.cam_mat(1, 2);
  int data_block_num = 0;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if ((cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)){
        continue;
      }
      // Add Data Term
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
          = vertex_set_[frm_idx].FindkNearestVertex(w, h, kNearestPoints + 1);
      std::vector<double*> parameter_blocks;
      int idx_k = h*kCamWidth + w;
      double img_obs = (double)(cam_set_[frm_idx].img_obs.at<uchar>(h, w));
      Eigen::Matrix<double, 3, 1> vec_M = calib_set_.M.block<3, 1>(0, idx_k);
      Eigen::Matrix<double, 3, 1> vec_D = calib_set_.D;
      double epi_A = epi_A_mat_.at<double>(h, w);
      double epi_B = epi_B_mat_.at<double>(h, w);
      DeformConstraint::DeformCostFunction* cost_function =
          DeformConstraint::Create(
              *pat_grid_, img_obs, vec_M, vec_D, epi_A, epi_B, fx, fy, dx, dy,
              calib_set_.light_vec_, kNearestPoints, frm_idx, vertex_nbr,
              data_set, frm_num_set, &parameter_blocks);
      if (cost_function != nullptr) {
        problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);
        data_block_num++;
      }
    }
  }

  // Add residual block: Regular Term
  int reg_block_num = 0;
  for (int ver_idx = 0; ver_idx < vertex_set_[frm_idx].len_; ver_idx++) {
    int w = vertex_set_[frm_idx].pos_(ver_idx, 0);
    int h = vertex_set_[frm_idx].pos_(ver_idx, 1);
    if (vertex_set_[frm_idx].valid_(ver_idx) != my::VERIFIED_TRUE)
      continue;
    Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
        = vertex_set_[frm_idx].FindkNearestVertex(w, h, kRegularNbr + 2);
    std::vector<double*> parameter_blocks;
    RegularConstraint::RegularCostFunction* cost_function =
        RegularConstraint::Create(
            alpha_val, alpha_norm, kRegularNbr, frm_idx,
            vertex_nbr, data_set, frm_num_set, &parameter_blocks);
    if (cost_function != nullptr) {
      problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);
      reg_block_num++;
    }
  }

  ceres::Solver::Options options;
  options.gradient_tolerance = 1e-10;
  options.function_tolerance = 1e-10;
  options.min_relative_decrease = 1e-1;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;

//  if (frm_idx == 3) {
//    int idx = vertex_set_[frm_idx].GetVertexIdxByPos(735, 825);
//    std::cout << "Before Opt: " << idx << std::endl;
//    std::cout << vertex_set_[frm_idx].vertex_val_(idx) << std::endl;
//    std::cout << (int)vertex_set_[frm_idx].valid_(idx) << std::endl;
//  }

  if (data_block_num == 0 && reg_block_num == 0) {
    return status;
  }
  ceres::Solve(options, &problem, &summary);

  // Optimization part may change the norm value. Need to be re-normalized.
  for (int ver_idx = 0; ver_idx < vertex_set_[frm_idx].len_; ver_idx++) {
    Eigen::Matrix<double, 1, 3> tmp_norm
        = vertex_set_[frm_idx].vertex_val_.block<1, 3>(ver_idx, 1);
    vertex_set_[frm_idx].vertex_val_.block(ver_idx, 1, 1, 3) = tmp_norm / tmp_norm.norm();
  }

//  if (frm_idx == 3) {
//    int idx = vertex_set_[frm_idx].GetVertexIdxByPos(735, 825);
//    std::cout << "After Opt: " << idx << std::endl;
//    std::cout << vertex_set_[frm_idx].vertex_val_(idx) << std::endl;
//    std::cout << (int)vertex_set_[frm_idx].valid_(idx) << std::endl;
//  }

  // Save?
//  std::cout << summary.FullReport() << std::endl;
  std::cout << "Data_Block: " << data_block_num << std::endl;
  std::cout << "Reg_Block: " << reg_block_num << std::endl;
  std::cout << summary.BriefReport() << std::endl;
  return status;
}

bool Reconstructor::CalculateDepthMat(int frm_idx) {
  bool status = true;

  cam_set_[frm_idx].depth.create(kCamHeight, kCamWidth, CV_64FC1);
  cam_set_[frm_idx].depth.setTo(0);
  cam_set_[frm_idx].norm_vec
      = Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, kCamHeight * kCamWidth);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
          = vertex_set_[frm_idx].FindkNearestVertex(w, h, kNearestPoints + 1);
      // Calculate w(i)*Z
      Eigen::Matrix<double, Eigen::Dynamic, 1> weight
          = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(kNearestPoints, 1);
      double sum_val = 0;
      for (int i = 0; i < kNearestPoints; i++) {
        weight(i) = pow((1 - vertex_nbr(i, 1) / vertex_nbr(kNearestPoints, 1)), 2);
        sum_val += weight(i);
      }
      // Calculate d_k
      double d_k = 0;
      for (int i = 0; i < kNearestPoints; i++) {
        int idx = (int)vertex_nbr(i, 0);
        d_k += (vertex_set_[frm_idx].vertex_val_(idx, 0) * weight(i)) / sum_val;
      }
      // Calculate norm
      Eigen::Vector3d norm_k = Eigen::Vector3d::Zero();
      for (int i = 0; i < kNearestPoints; i++) {
        int idx = (int)vertex_nbr(i, 0);
        Eigen::Matrix<double, 1, 3> tmp_norm
            = vertex_set_[frm_idx].vertex_val_.block<1, 3>(idx, 1);
        norm_k += weight(i) / sum_val * tmp_norm.transpose();
//        norm_k += tmp_norm.transpose();
//        break;
      }
//      std::cout << norm_k << std::endl;
      cam_set_[frm_idx].norm_vec.block<3, 1>(0, h * kCamWidth + w) = norm_k / norm_k.norm();
      cam_set_[frm_idx].depth.at<double>(h, w) = d_k;
//      if ((d_k >= kDepthMin) && (d_k <= kDepthMax)) {
//        cam_set_[frm_idx].depth.at<double>(h, w) = d_k;
//      } else {
//        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::MARKED;
//      }
      if (h == 885 && w == 705) {
//        std::cout << nb_idx_set << std::endl;
//        printf("d_k = %f\n", d_k);
//        printf("vertex_val_ul[%d] = %f\n", ul_idx, v_ul);
      }
    }
  }
  // Interpolation
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::MARKED) {
//        int search_rad = 5;
//        double sum_val = 0;
//        int sum_num = 0;
//        for (int d_h = -search_rad; d_h <= search_rad; d_h++) {
//          for (int d_w = -search_rad; d_w <= search_rad; d_w++) {
//            if (cam_set_[frm_idx].mask.at<uchar>(h+d_h, w+d_w) == my::VERIFIED_TRUE) {
//              sum_num++;
//              sum_val += cam_set_[frm_idx].depth.at<double>(h+d_h, w+d_w);
//            }
//          }
//        }
//        cam_set_[frm_idx].depth.at<double>(h, w) = sum_val / double(sum_num);
//        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_TRUE;
//      }
//    }
//  }
//  cv::namedWindow("depth" + Num2Str(frm_idx));
//  cv::imshow("depth" + Num2Str(frm_idx), cam_set_[frm_idx].depth / 30);
//  cv::waitKey(0);
//  cv::destroyWindow("depth" + Num2Str(frm_idx));
  return status;
}

bool Reconstructor::GenerateIest(int frm_idx) {
  bool status = true;

  cam_set_[frm_idx].img_est.create(kCamHeight, kCamWidth, CV_8UC1);
  cam_set_[frm_idx].img_est.setTo(0);
  cam_set_[frm_idx].shade_mat.create(kCamHeight, kCamWidth, CV_64FC1);
  cam_set_[frm_idx].shade_mat.setTo(0);
  cv::Mat norm_mat;
  norm_mat.create(kCamHeight, kCamWidth, CV_8UC1);
  norm_mat.setTo(0);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      // Get depth
      double depth = cam_set_[frm_idx].depth.at<double>(h, w);
      if (depth <= 0)
        continue;
      double x_pro = GetXproFromDepth(depth, h, w, calib_set_);
      double y_pro = GetYproFromXpro(x_pro, h, w, epi_A_mat_, epi_B_mat_);
      // Get img_intensity
      double img_k_head;
      pat_grid_->Evaluate(y_pro, x_pro, &img_k_head);
      // Set norm_weight
      Eigen::Vector3d norm_vec
          = cam_set_[frm_idx].norm_vec.block<3, 1>(0, h * kCamWidth + w);
      double norm_weight = norm_vec.transpose() * calib_set_.light_vec_;
      if ((norm_weight < 0) || (norm_weight > 1.0)) {
        if (norm_weight > 1.0)
          std::cout << "Error: " << norm_weight << std::endl
                    << norm_vec << std::endl;
        cam_set_[frm_idx].img_est.at<uchar>(h, w) = kMaskIntensityThred;
        continue;
      }
      cam_set_[frm_idx].shade_mat.at<double>(h, w) = norm_weight;
      cam_set_[frm_idx].img_est.at<uchar>(h, w) = uchar(img_k_head * norm_weight);
      norm_mat.at<uchar>(h, w) = uchar(255 * norm_weight);
    }
  }
  cv::namedWindow("I_est" + Num2Str(frm_idx));
  cv::imshow("I_est" + Num2Str(frm_idx), cam_set_[frm_idx].img_est);
  cv::waitKey(2000);
  cv::destroyWindow("I_est" + Num2Str(frm_idx));

  std::string norm_mat_name = output_file_path_ + "I_norm" + Num2Str(frm_idx) + ".png";
  cv::imwrite(norm_mat_name, norm_mat);
  return status;
}

bool Reconstructor::FixImageProbability(int frm_idx) {
  bool status = true;

  double R = 2.0;
  double Q = 1.0;
  if (frm_idx == 0) {
    cam_set_[frm_idx].P.create(kCamHeight, kCamWidth, CV_64FC1);
    cam_set_[frm_idx].P.setTo(2.0);
  } else {
    cam_set_[frm_idx].P = cam_set_[frm_idx - 1].P + R;
    for (int h = 0; h < kCamHeight; h++) {
      for (int w = 0; w < kCamWidth; w++) {
        double i_obs = (double)cam_set_[frm_idx].img_obs.at<uchar>(h, w);
        double i_est = (double)cam_set_[frm_idx].img_est.at<uchar>(h, w);
        double d_k = cam_set_[frm_idx].depth.at<double>(h, w);
        double P_k = cam_set_[frm_idx].P.at<double>(h, w);
        double L_k = i_est / d_k;
        double K = P_k*L_k / (L_k*P_k*L_k + Q);
        cam_set_[frm_idx].P.at<double>(h, w) = (1 - K*L_k) * P_k;
        cam_set_[frm_idx].depth.at<double>(h, w) = d_k + K*(i_obs - i_est);
      }
    }
  }

  return status;
}

bool Reconstructor::WriteResult(int frm_idx) {
  bool status = true;

//  printf("Write:\n");
//  printf("depth(%d, %d) = %f\n", 240, 105, cam_set_[frm_idx].depth.at<double>(240, 105));
//  printf("vex_val[%d] = %f\n", 1383, vertex_set_[frm_idx].vertex_val_(1383));

  std::string depth_png_name = output_file_path_ + depth_file_path_
                               + depth_file_name_ + Num2Str(frm_idx) + ".png";
  cv::imwrite(depth_png_name, (cam_set_[frm_idx].depth) / 30);
  std::string I_est_name = output_file_path_ + "I_est" + Num2Str(frm_idx) + ".png";
  cv::imwrite(I_est_name, cam_set_[frm_idx].img_est);
  std::string depth_txt_name = output_file_path_ + depth_file_path_
                                + depth_file_name_ + Num2Str(frm_idx) + ".txt";
  if (status)
    status = SaveMatToTxt(depth_txt_name, cam_set_[frm_idx].depth);

  std::string vertex_file_name = output_file_path_ + vertex_file_path_
                                 + vertex_file_name_ + Num2Str(frm_idx) + ".txt";
//  if (frm_idx == 3) {
//    int idx = vertex_set_[frm_idx].GetVertexIdxByPos(735, 825);
//    std::cout << "When Write: " << idx << std::endl;
//    std::cout << vertex_set_[frm_idx].vertex_val_(idx) << std::endl;
//    std::cout << (int)vertex_set_[frm_idx].valid_(idx) << std::endl;
//  }
  if (status)
    status = SaveValToTxt(
        vertex_file_name, vertex_set_[frm_idx].vertex_val_,
        vertex_set_[frm_idx].block_height_, vertex_set_[frm_idx].block_width_);
  std::string valid_file_name = output_file_path_ + valid_file_path_
                                + valid_file_name_ + Num2Str(frm_idx) + ".txt";
  if (status)
    status = SaveVecUcharToTxt(valid_file_name, vertex_set_[frm_idx].valid_,
                               vertex_set_[frm_idx].block_height_,
                               vertex_set_[frm_idx].block_width_);
  return status;
}

 */
