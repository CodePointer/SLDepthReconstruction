//
// Created by pointer on 17-12-15.
//


#include <depth_spacial_constraint.h>
#include <depth_temporal_constraint.h>
#include <depth_consist_constraint.h>
#include "reconstructor.h"

Reconstructor::Reconstructor() {
  cam_set_ = nullptr;
//  cam_slots_ = nullptr;
  node_set_ = nullptr;
//  vertex_set_ = nullptr;
  pat_grid_ = nullptr;
}

Reconstructor::~Reconstructor() {
  if (cam_set_ != nullptr) {
    delete[]cam_set_;
    cam_set_ = nullptr;
  }
//  if (cam_slots_ != nullptr) {
//    delete cam_slots_;
//    cam_slots_ = nullptr;
//  }
  if (node_set_ != nullptr) {
    delete[]node_set_;
    node_set_ = nullptr;
  }
//  if (vertex_set_ != nullptr) {
//    delete[]vertex_set_;
//    vertex_set_ = nullptr;
//  }
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
  // Used for input and output.
  kFrameNum = 300;
  main_file_path_ = "/home/pointer/CLionProjects/Data/20180624/HandMove_BlkDis/";
//  pattern_file_name_ = "pattern_4size4color0.png";
//  class_file_name_ = "class_4size4color0.png";
//  pattern_file_suffix_ = ".png";
  dyna_file_path_ = "dyna/";
  dyna_file_name_ = "dyna_mat";
  dyna_file_suffix_ = ".png";
  pro_file_path_ = "pro/";
  pro_file_name_ = "xpro_mat";
  pro_file_suffix_ = ".txt";
  // epi_A_file_name_ = "EpiMatA.txt";
  // epi_B_file_name_ = "EpiMatB.txt";
  cam_matrix_name_ = "cam_mat0.txt";
  pro_matrix_name_ = "pro_mat0.txt";
  rots_name_ = "R0.txt";
  trans_name_ = "T0.txt";
  // light_name_ = "light_vec.txt";
  // hard_mask_name_ = "hard_mask.png";
  //  hard_mask_file_name_ = "cam0_pro/hard_mask.png";
  // Set output file path
  output_file_path_ = main_file_path_ + "result/";
  // depth_file_path_ = "";
  // depth_file_name_ = "depth";

  // Load Informations
  // pattern_lower_ = 30.0;
  // pattern_higher_ = 180.0;
  status = LoadDatasFromFiles();

  // Set pattern grid
  // cv::namedWindow("pattern");
  // cv::imshow("pattern", pattern_ / 255.0);
  // cv::waitKey(0);
  // cv::destroyWindow("pattern");
  LOG(INFO) << "Reconstructor::Init finished.";
  return status;
}

bool Reconstructor::LoadDatasFromFiles() {
  LOG(INFO) << "Loading data from hard disk.";
  ///-----------------------------------///
  /// Calib_Set:
  /// load cam_matrix & pro_matrix,
  /// set M&D
  ///-----------------------------------///
  LoadTxtToEigen(main_file_path_ + cam_matrix_name_, 3, 3, calib_set_.cam.data());
  LoadTxtToEigen(main_file_path_ + pro_matrix_name_, 3, 3, calib_set_.pro.data());
  LoadTxtToEigen(main_file_path_ + rots_name_, 3, 3, calib_set_.R.data());
  LoadTxtToEigen(main_file_path_ + trans_name_, 3, 1, calib_set_.t.data());
//  LoadTxtToEigen(main_file_path_ + light_name_, 3, 1, calib_set_.light_vec_.data());
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

  // Cam_set
  cam_set_ = new CamMatSet[kFrameNum];
//  cam_slots_ = new CamSlotsMat;
  node_set_ = new NodeSet[kFrameNum];
//  vertex_set_ = new VertexSet[kFrameNum];
  for (int frm_idx = 0; frm_idx < kFrameNum; frm_idx++) {
    cam_set_[frm_idx].img_obs = cv::imread(main_file_path_
                                           + dyna_file_path_
                                           + dyna_file_name_
                                           + Num2Str(frm_idx)
                                           + dyna_file_suffix_,
                                           cv::IMREAD_COLOR);
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

/// ConvXpro2Depth: xpro_mat -> depth_mat.
/// Need: mask, x_pro
/// Create: depth
/// Change: <null>
/// Without interpolation.
void Reconstructor::ConvXpro2Depth(int frm_idx, bool create_flag) {
  LOG(INFO) << "Start: ConvXpro2Depth(" << frm_idx << ")";
  if (create_flag) {
    cam_set_[frm_idx].depth.create(kCamHeight, kCamWidth, CV_64FC1);
    cam_set_[frm_idx].depth.setTo(-1.0);
  }
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      double x_pro = cam_set_[frm_idx].x_pro.at<double>(h, w);
      if (x_pro < 0) {
        continue;
        // WARNING: Invalid depth mask
      }
      double depth = GetDepthFromXpro(x_pro, h, w, &calib_set_);
      cam_set_[frm_idx].depth.at<double>(h, w) = depth;
    }
  }
//  cv::namedWindow("test");
//  cv::imshow("test", (ptr_cam_set->depth/100));
//  cv::waitKey(0);
  LOG(INFO) << "End: ConvXpro2Depth(" << frm_idx << ")";
}

/// ConvXpro2Depth: xpro_mat -> depth_mat.
/// Need: mask, depth
/// Create: x_pro (create_flag=true)
/// Change: <null>
/// Without interpolation.
void Reconstructor::ConvDepth2Xpro(int frm_idx, bool create_flag) {
  LOG(INFO) << "Start: ConvDepth2Xpro(" << frm_idx << ")";
  if (create_flag) {
    cam_set_[frm_idx].x_pro.create(kCamHeight, kCamWidth, CV_64FC1);
    cam_set_[frm_idx].x_pro.setTo(-1.0);
  }
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      double depth = cam_set_[frm_idx].depth.at<double>(h, w);
      if (depth <= 0) {
        continue;
      }
      double x_pro = GetXproFromDepth(depth, h, w, &calib_set_);
      cam_set_[frm_idx].x_pro.at<double>(h, w) = x_pro;
    }
  }
//  cv::namedWindow("test");
//  cv::imshow("test", (ptr_cam_set->depth/100));
//  cv::waitKey(0);
  LOG(INFO) << "End: ConvDepth2Xpro(" << frm_idx << ")";
}

/// FillDepthWithMask: fill depth_mat/pro_mat for every point in mask.
/// Need: mask, depth/x_pro
/// Create: <null>
/// Change: depth/x_pro
/// Use expanded methods. mat <= 0 -> invalid points.
void Reconstructor::FillMatWithMask(cv::Mat * ptr_mask, cv::Mat * ptr_mat) {
  LOG(INFO) << "Start: FillMatWithMask";
  bool expand_flag = true;
  cv::Mat tmp_value_mat;
  ptr_mat->copyTo(tmp_value_mat);
  int found_num = 0;
  while (expand_flag) {
    expand_flag = false;
    for (int h = 0; h < kCamHeight; h++) {
      for (int w = 0; w < kCamWidth; w++) {
        if (ptr_mask->at<uchar>(h, w) != my::VERIFIED_TRUE)
          continue;
        if (ptr_mat->at<double>(h, w) > 0) {
          int h_n[4] = {h-1, h, h+1, h};
          int w_n[4] = {w, w-1, w, w+1};
          for (int i = 0; i < 4; i++) {
            if (ptr_mat->at<double>(h_n[i], w_n[i]) <= 0) {
              expand_flag = true;
              tmp_value_mat.at<double>(h_n[i], w_n[i])
                  = ptr_mat->at<double>(h, w);
              found_num += 1;
            }
          }
        }
      }
    }
    tmp_value_mat.copyTo(*ptr_mat);
  }
  LOG(INFO) << "End: FillMatWithMask. Found num: " << found_num;
}

bool Reconstructor::Run() {
  bool status = true;

  cv::namedWindow("x_pro");
  LOG(INFO) << "Reconstruction process start.";

  ///------------------------------///
  /// For first frame (frame 0):
  ///------------------------------///
  LOG(INFO) << "First frame process start.";
  // For every point in mask, have x_pro value.
  SetMaskMatFromIobs(0);
  RecoColorClass(0);
  ConvXpro2Depth(0, true);
  FillMatWithMask(&(cam_set_[0].mask), &(cam_set_[0].depth));
  ConvDepth2Xpro(0, false);
  FillMatWithMask(&(cam_set_[0].mask), &(cam_set_[0].x_pro));
  LOG(INFO) << "First frame process finished.";

  ///------------------------------///
  /// For other frames (frame 1-N):
  ///------------------------------///
  for (int frm_idx = 1; (frm_idx < kFrameNum) && status; frm_idx++) {
    LOG(INFO) << "Frame (" << frm_idx << ") processing";
    std::cout << "Frame (" << frm_idx << "):" << std::endl;

    // Set mask for point.
    SetMaskMatFromIobs(frm_idx);
    RecoColorClass(frm_idx);

    // Predict x_pro, depth from last frame.
    PredictXproRange(frm_idx);
    ConvXpro2Depth(frm_idx, true);
//    FillMatWithMask(&(cam_set_[frm_idx].mask), &(cam_set_[frm_idx].depth));
//    ShowMat<ushort>(&cam_set_[frm_idx].x_pro_range, "pro_range", 0, 100, 1200);
//    ShowMat<double>(&cam_set_[frm_idx].depth, "DepthRange", 0, 10, 30);

    // Set Node mat and optimization
    SetNodeFromDepthVal(frm_idx);
    OptimizeDepthNode(frm_idx);

    // Fill other points of node mat (For next frame usage)
    SetDepthValFromNode(frm_idx);
//    ShowMat<double>(&(cam_set_[frm_idx].depth), "depth_before", 0, 15, 30);
    FillMatWithMask(&(cam_set_[frm_idx].mask), &(cam_set_[frm_idx].depth));
//    ShowMat<double>(&(cam_set_[frm_idx].depth), "depth_after", 0, 15, 30);
    ConvDepth2Xpro(frm_idx, false);
//    FillMatWithMask(&(cam_set_[frm_idx].mask), &(cam_set_[frm_idx].x_pro));

    ShowMat<double>(&cam_set_[frm_idx].x_pro, "x_pro", 10, 200, 1000, false);

    if (frm_idx - kTemporalWindowSize > 1) {
      SetDepthValFromNode(frm_idx - kTemporalWindowSize - 1);
      ConvDepth2Xpro(frm_idx - kTemporalWindowSize - 1, false);
      std::cout << "Writing & Releasing frm: "
                << frm_idx - kTemporalWindowSize - 1 << std::endl;
      OutputResult(frm_idx - kTemporalWindowSize - 1);
      ReleaseSpace(frm_idx - kTemporalWindowSize - 1);
    }
    LOG(INFO) << "Frame " << frm_idx << "finished.";
  }
  return status;
}

// Set Mask mat according to pixel from I_obs
void Reconstructor::SetMaskMatFromIobs(int frm_idx) {
  LOG(INFO) << "Start: SetMaskMatFromIobs(" << frm_idx << ");";
  cam_set_[frm_idx].mask.create(kCamHeight, kCamWidth, CV_8UC1);
  std::string mask_file_path = main_file_path_ + "mask_res/";
  std::string mask_file_names = "mask" + Num2Str(frm_idx) + ".png";
  cv::Mat tmp = cv::imread(mask_file_path + mask_file_names, cv::IMREAD_UNCHANGED);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (tmp.at<uchar>(h, w) == 0) {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_FALSE;
      } else {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_TRUE;
      }
    }
  }
  // Set mask initial value by intensity:
  //   my::INITIAL_FALSE for dark part
  //   my::INITIAL_TRUE for light part
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if (cam_set_[frm_idx].img_obs.at<uchar>(h, w) <= kMaskIntensityThred) {
//        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::INITIAL_FALSE;
//      } else {
//        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::INITIAL_TRUE;
//      }
//    }
//  }
  // FloodFill every point
  //   For my::INITIAL_FALSE:
  //     area < Thred: Should be set to my::VERIFIED_TRUE
  //   for my::INITIAL_TRUE:
  //     area < Thred: Should be set to my::VERIFIED_FALSE
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      uchar val = cam_set_[frm_idx].mask.at<uchar>(h, w);
//      if (val == my::INITIAL_FALSE) {
//        int num = FloodFill(cam_set_[frm_idx].mask, h, w,
//                            my::MARKED, my::INITIAL_FALSE);
//        if (num <= kMaskMinAreaThred) {
//          FloodFill(cam_set_[frm_idx].mask, h, w,
//                    my::VERIFIED_TRUE, my::MARKED);
//        } else {
//          FloodFill(cam_set_[frm_idx].mask, h, w,
//                    my::VERIFIED_FALSE, my::MARKED);
//        }
//      } else if (val == my::INITIAL_TRUE) {
//        int num = FloodFill(cam_set_[frm_idx].mask, h, w,
//                            my::MARKED, my::INITIAL_TRUE);
//        if (num <= kMaskMinAreaThred) {
//          FloodFill(cam_set_[frm_idx].mask, h, w,
//                    my::VERIFIED_FALSE, my::MARKED);
//        } else {
//          FloodFill(cam_set_[frm_idx].mask, h, w,
//                    my::VERIFIED_TRUE, my::MARKED);
//        }
//      }
//    }
//  }
  // Set result with hard_mask
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if ((hard_mask_.at<uchar>(h, w) == my::VERIFIED_FALSE)
//          && (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_TRUE)) {
//        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_FALSE;
//      }
//    }
//  }
//  cv::imshow("test", cam_set_[frm_idx].mask * 255);
//  cv::waitKey(0);

  LOG(INFO) << "End: SetMaskMatFromIobs(" << frm_idx << ");";
}

/// RecoColorClass: Classify intensity class.
/// Need: mask, img_obs
/// Create: img_class
/// Change: <null>
/// Use K-means. Now is just load from files.
void Reconstructor::RecoColorClass(int frm_idx) {
  LOG(INFO) << "Start: RecoColorClass(" << frm_idx << ")";

  cam_set_[frm_idx].img_class.create(kCamHeight, kCamWidth, CV_8UC1);
  cam_set_[frm_idx].img_class.setTo(0);
  std::string class_file_path = main_file_path_ + "class_res/";
  std::string class_file_names = "class" + Num2Str(frm_idx) + ".png";
  cv::Mat tmp = cv::imread(class_file_path + class_file_names, cv::IMREAD_UNCHANGED);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      if (tmp.at<uchar>(h, w) == 0) {
        LOG(ERROR) << "Invalid class at: " << h << "," << w;
      } else {
        uchar value = tmp.at<uchar>(h, w);
        cam_set_[frm_idx].img_class.at<uchar>(h, w) = value / uchar(10);
      }
    }
  }
  LOG(INFO) << "End: RecoColorClass(" << frm_idx << ")";
}

/// PredictXproRange: predict initial x_pro_range from history.
/// Need: mask[t], img_class[t], x_pro[t-1]
/// Create: x_pro_range[t], x_pro[t]
/// Change: mask
/// By last frame average x_pro value. 7*7 for now.
void Reconstructor::PredictXproRange(int frm_idx) {
  LOG(INFO) << "Start: PredictXproRange(" << frm_idx << ")";
  cam_set_[frm_idx].x_pro_range.create(kCamHeight, kCamWidth, CV_16UC1);
  cam_set_[frm_idx].x_pro_range.setTo(0);
  cam_set_[frm_idx].x_pro.create(kCamHeight, kCamWidth, CV_64FC1);
  cam_set_[frm_idx].x_pro.setTo(-1);

  // Propagate last frame info to current frame
  cv::Mat last_xpro_info(kCamHeight, kCamWidth, CV_64FC1);
  last_xpro_info.setTo(-1);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      if (cam_set_[frm_idx - 1].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      last_xpro_info.at<double>(h, w)
          = cam_set_[frm_idx - 1].x_pro.at<double>(h, w);
    }
  }
  FillMatWithMask(&(cam_set_[frm_idx].mask), &last_xpro_info);

  int kWinRad = 7;
  double kStripDis = 12;
  double kClassNum = 6;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      // Get last x_pro value
      double last_x_pro_value = 0;
      int last_x_pro_count = 0;
      for (int nh = h - kWinRad; nh <= h + kWinRad; nh++) {
        for (int nw = w - kWinRad; nw <= w + kWinRad; nw++) {
          if (nh < 0 || nh >= kCamHeight || nw < 0 || nw >= kCamWidth)
            continue;
          if (last_xpro_info.at<double>(nh, nw) < 0)
            continue;
          last_x_pro_value += last_xpro_info.at<double>(nh, nw);
          last_x_pro_count += 1;
        }
      }
      last_x_pro_value /= last_x_pro_count;
      // Check period of now
      int class_num = 7 - cam_set_[frm_idx].img_class.at<uchar>(h, w);
      double left_idx = std::floor((last_x_pro_value - kStripDis * class_num)
                                   / (kStripDis * kClassNum))
                        * kStripDis * kClassNum
                        + kStripDis * class_num;
      double right_idx = std::ceil((last_x_pro_value - kStripDis * class_num)
                                   / (kStripDis * kClassNum))
                         * kStripDis * kClassNum
                         + kStripDis * class_num;
      if (std::abs(last_x_pro_value - left_idx)
          <= std::abs(right_idx - last_x_pro_value))
        cam_set_[frm_idx].x_pro_range.at<ushort>(h, w) = (ushort)left_idx;
      else
        cam_set_[frm_idx].x_pro_range.at<ushort>(h, w) = (ushort)right_idx;
      // Set x_pro from x_pro_range
      cam_set_[frm_idx].x_pro.at<double>(h, w)
          = (double)cam_set_[frm_idx].x_pro_range.at<ushort>(h, w);
      if (cam_set_[frm_idx].x_pro_range.at<ushort>(h, w) == 0) {
//        std::cout << "Error: " << h << "," << w << std::endl;
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_FALSE;
      }
    }
  }
  LOG(INFO) << "End: PredictXproRange(" << frm_idx << ")";
}

/// OptimizeDepthNode: Optimize Node mat by spatial & temporal info
/// \Need: node[t] x_pro_range[t]
/// \Create: <null>
/// \Change: node[t]
/// Use spatial & temporal constraint.
/// Also, set upper-bound & lower-bound for optimization.
void Reconstructor::OptimizeDepthNode(int frm_idx) {
  LOG(INFO) << "Start: OptimizeDepthNode(" << frm_idx << ")";

  // parameters:
  ceres::Problem problem;
  int spatial_block_num = 0;
  int temporal_block_num = 0;
  int consist_block_num = 0;

  // For history:
  int t_start = (frm_idx - kTemporalWindowSize >= 1) ? frm_idx - kTemporalWindowSize : 1;
  int t_now = frm_idx;
  int block_num = 0;
  for (int t_cen = t_now; t_cen >= t_start; t_cen--) {
    for (int idx_cen = 0; idx_cen < node_set_[t_cen].len_; idx_cen++) {
      if (node_set_[t_cen].valid_(idx_cen, 0) != my::VERIFIED_TRUE)
        continue;
      int h_cen, w_cen;
      node_set_[t_cen].GetNodeCoordByIdx(idx_cen, &h_cen, &w_cen);
      int pos_x = node_set_[t_cen].pos_(idx_cen, 0);
      int pos_y = node_set_[t_cen].pos_(idx_cen, 1);

      // Add space block: depth gradient, vertical & horizenal
      std::vector<double *> parameter_blocks;
      std::vector<int> idx_list;
      DepthSpacialConstraint::DepthSpacialCostFunction *spatical_cost_function
          = DepthSpacialConstraint::Create(
              Num2Str(t_cen) + " " + Num2Str(idx_cen),
              &node_set_[t_cen], idx_cen, true, &idx_list, &parameter_blocks);
      if (spatical_cost_function != nullptr) {
        problem.AddResidualBlock(spatical_cost_function, nullptr,
                                 parameter_blocks);
        spatial_block_num++;
        for (int i = 0; i < idx_list.size(); i++) {
          int idx = idx_list[i];
          problem.SetParameterLowerBound(
              parameter_blocks[i], 0, node_set_[t_cen].bound_(idx, 0));
          problem.SetParameterUpperBound(
              parameter_blocks[i], 0, node_set_[t_cen].bound_(idx, 1));
        }
      }
      spatical_cost_function
          = DepthSpacialConstraint::Create(
          Num2Str(t_cen) + " " + Num2Str(idx_cen),
          &node_set_[t_cen], idx_cen, false, &idx_list, &parameter_blocks);
      if (spatical_cost_function != nullptr) {
        problem.AddResidualBlock(spatical_cost_function, nullptr,
                                 parameter_blocks);
        spatial_block_num++;
        for (int i = 0; i < idx_list.size(); i++) {
          int idx = idx_list[i];
          problem.SetParameterLowerBound(
              parameter_blocks[i], 0, node_set_[t_cen].bound_(idx, 0));
          problem.SetParameterUpperBound(
              parameter_blocks[i], 0, node_set_[t_cen].bound_(idx, 1));
        }
      }

      // Add temporal block: depth speed
      if (t_cen - 2 >= t_start) {
        DepthTemporalConstraint::DepthTemporalCostFunction *
            temporal_cost_funciton
            = DepthTemporalConstraint::Create(
                Num2Str(t_cen) + " " + Num2Str(idx_cen), idx_cen,
                &node_set_[t_cen], &node_set_[t_cen - 1], &node_set_[t_cen - 2],
                &parameter_blocks);
        if (temporal_cost_funciton != nullptr) {
          problem.AddResidualBlock(temporal_cost_funciton, nullptr,
                                   parameter_blocks);
          temporal_block_num++;
          for (int i = 0; i <= 2; i++) {
            problem.SetParameterLowerBound(
                parameter_blocks[i], 0,
                node_set_[t_cen - i].bound_(idx_cen, 0));
            problem.SetParameterUpperBound(
                parameter_blocks[i], 0,
                node_set_[t_cen - i].bound_(idx_cen, 1));
          }
        }
      }

      // Add temporal consistent block
      if (t_cen == t_start + 1 && t_start >= 2) {
        DepthConsistConstraint::DepthCosistCostFunction *
            consist_cost_function
            = DepthConsistConstraint::Create(
                Num2Str(t_cen) + " " + Num2Str(idx_cen), idx_cen,
                &node_set_[t_cen], &node_set_[t_cen - 1], &node_set_[t_cen - 2],
                &parameter_blocks);
        if (consist_cost_function != nullptr) {
          problem.AddResidualBlock(consist_cost_function, nullptr,
                                   parameter_blocks);
          consist_block_num++;
          for (int i = 0; i <= 1; i++) {
            problem.SetParameterLowerBound(
                parameter_blocks[i], 0,
                node_set_[t_cen - i].bound_(idx_cen, 0));
            problem.SetParameterUpperBound(
                parameter_blocks[i], 0,
                node_set_[t_cen - i].bound_(idx_cen, 1));
          }
        }
      }
    }
  }

  // Solve
  ceres::Solver::Options options;
//  options.gradient_tolerance = 1e-10;
//  options.function_tolerance = 1e-10;
//  options.min_relative_decrease = 1e1;
//  options.parameter_tolerance = 1e-20;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;

  LOG(INFO) << "Stat: OptimizeDepthNode(" << frm_idx << "):"
            << spatial_block_num << "," << temporal_block_num << ","
            << consist_block_num;
  std::cout << "Start optimization: " << frm_idx << std::endl;
  std::cout << "\tSpatial_block = " << spatial_block_num << std::endl;
  std::cout << "\tTemporal_block = " << temporal_block_num << std::endl;
  std::cout << "\tConsist_block = " << consist_block_num << std::endl;
  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << summary.BriefReport();
  std::cout << summary.BriefReport() << std::endl;
  std::cout << summary.message << std::endl;
  LOG(INFO) << "End: OptimizeDepthNode(" << frm_idx << ")";
}

/// SetNodeFromDepthVal: Set node[t] from depth[t], x_pro_range[t]
/// \Need: mask[t], img_class[t], depth[t]
/// \Create: node[t]
/// \Change: <null>
/// Select valid point for node
/// Also set depth range
void Reconstructor::SetNodeFromDepthVal(int frm_idx) {
  LOG(INFO) << "Start: SetNodeFromDepthVal(" << frm_idx << ")";
  double kStripDis = 12;
  for (int i = 0; i < node_set_[frm_idx].len_; i++) {
    int h_ord, w_ord;
    node_set_[frm_idx].GetNodeCoordByIdx(i, &h_ord, &w_ord);
    int h = h_ord * kNodeBlockSize;
    int w = w_ord * kNodeBlockSize;

    // Set pos & val: find center_point & 3*3 same class pixel
    int pos_x = -1, pos_y = -1;
    double min_dis = 1000.0;
    double center_x = w + ((double)kNodeBlockSize - 1.0) / 2;
    double center_y = h + ((double)kNodeBlockSize - 1.0) / 2;
    for (int y = h + 1; y < h + kNodeBlockSize - 1; y++) {
      for (int x = w + 1; x < w + kNodeBlockSize - 1; x++) {
        if (cam_set_[frm_idx].mask.at<uchar>(y, x) != my::VERIFIED_TRUE)
          continue;
        bool same_flag = true;
        for (int ny = y - 2; (ny <= y + 2) && same_flag; ny++) {
          for (int nx = x - 2; (nx <= x + 2) && same_flag; nx++) {
            if (cam_set_[frm_idx].img_class.at<uchar>(y, x)
                != cam_set_[frm_idx].img_class.at<uchar>(ny, nx)) {
              same_flag = false;
              break;
            }
          }
        }
        if (same_flag) {
          double dis = pow(x - center_x, 2) + pow(y - center_y, 2);
          if (dis < min_dis) {
            pos_x = x; pos_y = y;
            min_dis = dis;
          }
        }
      }
    }
    if (pos_x < 0 || pos_y < 0) {
      node_set_[frm_idx].valid_(i, 0) = my::VERIFIED_FALSE;
      continue;
    }

    node_set_[frm_idx].SetNodePos(i, pos_x, pos_y);
    double depth_val = cam_set_[frm_idx].depth.at<double>(pos_y, pos_x);
    node_set_[frm_idx].val_(i, 0) = depth_val;
    double x_pro_upper = cam_set_[frm_idx].x_pro_range.at<ushort>(pos_y, pos_x)
                         + kStripDis / 2;
    double x_pro_lower = cam_set_[frm_idx].x_pro_range.at<ushort>(pos_y, pos_x)
                         - kStripDis / 2;
    double depth_upper = GetDepthFromXpro(x_pro_upper, pos_y, pos_x, &calib_set_);
    double depth_lower = GetDepthFromXpro(x_pro_lower, pos_y, pos_x, &calib_set_);
    node_set_[frm_idx].bound_(i, 0) = depth_lower;
    node_set_[frm_idx].bound_(i, 1) = depth_upper;
    node_set_[frm_idx].valid_(i, 0) = my::VERIFIED_TRUE;
  }
  LOG(INFO) << "End: SetNodeFromDepthVal(" << frm_idx << ")";
}

/// SetDepthValFromNode: Set depth[t] from node[t]
/// \Need: mask[t], depth[t], x_pro_range[t]
/// \Create: <null>
/// \Change: depth[t]
/// Select valid point for node
void Reconstructor::SetDepthValFromNode(int frm_idx) {
  // Set coarse depth_val from node
  cam_set_[frm_idx].depth.setTo(-1);
  for (int i = 0; i < node_set_[frm_idx].len_; i++) {
    if (node_set_[frm_idx].valid_(i, 0) != my::VERIFIED_TRUE)
      continue;
    int h = node_set_[frm_idx].pos_(i, 1);
    int w = node_set_[frm_idx].pos_(i, 0);
    double depth = node_set_[frm_idx].val_(i, 0);
    cam_set_[frm_idx].depth.at<double>(h, w) = depth;
  }

  // Fill values
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      std::vector<int> res;
//      node_set_[frm_idx].GetNearestkNodesIdx(w, h, &res);
      node_set_[frm_idx].GetTriVertexIdx(w, h, &res);
      if (res.empty()) {
        continue;
      }
      double P1x = node_set_[frm_idx].pos_(res[0], 0);
      double P1y = node_set_[frm_idx].pos_(res[0], 1);
      double P2x = node_set_[frm_idx].pos_(res[1], 0);
      double P2y = node_set_[frm_idx].pos_(res[1], 1);
      double P3x = node_set_[frm_idx].pos_(res[2], 0);
      double P3y = node_set_[frm_idx].pos_(res[2], 1);
      Eigen::Matrix<double, 2, 2> A;
      Eigen::Matrix<double, 2, 1> B;
      A(0, 0) = P2x - P1x;
      A(0, 1) = P3x - P1x;
      A(1, 0) = P2y - P1y;
      A(1, 1) = P3y - P1y;
      A << P2x - P1x, P3x - P1x, P2y - P1y, P3y - P1y;
      B << w - P1x, h - P1y;
      Eigen::Matrix<double, 2, 1> C = A.inverse() * B;
      double u = C(0, 0);
      double v = C(1, 0);
//      u = std::abs(u) < 1e-4 ? 0 : u;
//      v = std::abs(v) < 1e-4 ? 0 : v;
      if (u >= 0 && u <= 1 && v >= 0 && v <= 1 && u+v >= 0 && u+v <= 1) {
        double D1 = cam_set_[frm_idx].depth.at<double>(P1y, P1x);
        double D2 = cam_set_[frm_idx].depth.at<double>(P2y, P2x);
        double D3 = cam_set_[frm_idx].depth.at<double>(P3y, P3x);
        double depth = (1-u-v) * D1 + u * D2 + v * D3;
        cam_set_[frm_idx].depth.at<double>(h, w) = depth;
      } else {
//        std::cout << h << "," << w << std::endl;
//        std::cout << cam_set_[frm_idx].depth.at<double>(h, w) << std::endl;
//        std::cout << "A:" << std::endl;
//        std::cout << A << std::endl;
//        std::cout << B << std::endl;
//        std::cout << C << std::endl;
//        LOG(FATAL) << "Invalid tri at:" << h << ", " << w;
//        std::cout << h << "," << w << std::endl;
        continue;
        // WARNING: Depth empty hole
      }
    }
  }
//  ShowMat<double>(&(cam_set_[frm_idx].depth), "depth", 0, 30, 35);

  // Create optimize problem to fill dense depth mat
//  ceres::Problem problem;
//  for (int h = 0; h < kCamHeight; h++) {
//    for (int w = 0; w < kCamWidth; w++) {
//      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
//        continue;
//      if (node_set_[frm_idx].IsNode(w, h)) {
//        continue;
//      }
//      double * depth_pointer = &(cam_set_[frm_idx].depth.at<double>(h, w));
//      std::string part_info = "[" + Num2Str(h) + "," + Num2Str(w) + "] -> ";
//      // Search for nbr:
//      for (int h_n = h - 1; h_n <= h + 1; h_n++) {
//        for (int w_n = w - 1; w_n <= w + 1; w_n++) {
//          if (h == h_n && w == w_n) {
//            continue;
//          }
//          if (h_n < 0 || h_n >= kCamHeight || w_n < 0 || w_n >= kCamWidth) {
//            continue;
//          }
//          if (cam_set_[frm_idx].mask.at<uchar>(h_n, w_n) != my::VERIFIED_TRUE) {
//            continue;
//          }
//          bool flag_fix = node_set_[frm_idx].IsNode(w_n, h_n);
//          double * nbr_pointer = &(cam_set_[frm_idx].depth.at<double>(h_n, w_n));
//          std::string info = part_info
//                             + "[" + Num2Str(h_n) + "," + Num2Str(w_n) + "]";
//          std::vector<double *> parameter_blocks;
//          DepthInterConstraint::DepthInterCostFunction *inter_cost_function
//              = DepthInterConstraint::Create(info, flag_fix, depth_pointer,
//                                             nbr_pointer, &parameter_blocks);
//          problem.AddResidualBlock(inter_cost_function, nullptr,
//                                   parameter_blocks);
//        }
//      }
//    }
//  }
//
//  ceres::Solver::Options solver_options;
//  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//  solver_options.minimizer_progress_to_stdout = true;
//  solver_options.min_relative_decrease = 1e-9;
//
//  ceres::Solver::Summary summary;
//  Solve(solver_options, &problem, &summary);
//  LOG(INFO) << summary.BriefReport();
//  std::cout << summary.BriefReport() << std::endl;
//  std::cout << summary.message << std::endl;

//  ShowMat<double>(&(cam_set_[frm_idx].depth), "initial_depth", 0, 30.0, 35.0);

}

// Output all the result for further analysis
void Reconstructor::OutputResult(int frm_idx) {
  LOG(INFO) << "Start: OutputResult(" << frm_idx << ")";
  bool status = true;
  // Depth mat
  std::string depth_txt_name
      = output_file_path_ + depth_file_path_ + "depth"
        + Num2Str(frm_idx) + ".txt";
  status = SaveMatToTxt(depth_txt_name, cam_set_[frm_idx].depth);
  if (status) LOG(INFO) << depth_txt_name << " success.";

  // x_pro mat
  std::string xpro_txt_name
      = output_file_path_ + "x_pro" + Num2Str(frm_idx) + ".txt";
  status = SaveMatToTxt(xpro_txt_name, cam_set_[frm_idx].x_pro);
  if (status) LOG(INFO) << xpro_txt_name << " success.";

  // x_pro_range
//  ShowMat<ushort>(&cam_set_[frm_idx].x_pro_range, "pro_range", 0, 100, 1200);
  std::string xpro_range_png_name
      = output_file_path_ + "x_pro_range" + Num2Str(frm_idx) + ".png";
  status = cv::imwrite(xpro_range_png_name, cam_set_[frm_idx].x_pro_range);
  if (status) LOG(INFO) << xpro_range_png_name << "success.";

//  cv::Mat depth_show;
//  std::string depth_png_name
//      = output_file_path_ + depth_file_name_ + Num2Str(frm_idx) + ".png";
//  ShowMat<double>(&(cam_set_[frm_idx].depth), "depth", 10, 30.0, 35.0, &depth_show);
//  status = cv::imwrite(depth_png_name, depth_show);
//  if (status) LOG(INFO) << depth_png_name << " success.";

//  // Shade mat
//  std::string shade_txt_name
//      = output_file_path_ + "shade" + Num2Str(frm_idx) + ".txt";
//  status = SaveMatToTxt(shade_txt_name, cam_set_[frm_idx].shade_mat);
//  if (status) LOG(INFO) << shade_txt_name << " success.";
//  // Shade mat show
//  std::string shade_png_name
//      = output_file_path_ + "shade" + Num2Str(frm_idx) + ".png";
//  status = cv::imwrite(shade_png_name, cam_set_[frm_idx].shade_mat * 255.0);
//  if (status) LOG(INFO) << shade_png_name << " success.";
//
//  // img_obs
//  std::string obs_img_name
//      = output_file_path_ + "I_obs" + Num2Str(frm_idx) + ".png";
//  status = cv::imwrite(obs_img_name, cam_set_[frm_idx].img_obs);
//  if (status) LOG(INFO) << obs_img_name << " success.";
//  // img_est
//  std::string est_img_name
//      = output_file_path_ + "I_est" + Num2Str(frm_idx) + ".png";
//  status = cv::imwrite(est_img_name, cam_set_[frm_idx].img_est);
//  if (status) LOG(INFO) << est_img_name << " success.";
  // pattern_class
//  std::string class_img_name
//      = output_file_path_ + "I_class" + Num2Str(frm_idx) + ".png";
//  status = cv::imwrite(class_img_name, cam_set_[frm_idx].img_class);
//  if (status) LOG(INFO) << class_img_name << " success.";

//  std::string class_img_p_name
//      = output_file_path_ + "I_class_p" + Num2Str(frm_idx) + ".png";
//  cv::Mat class_p;
//  ShowMat<double>(&(cam_set_[frm_idx].img_class_p), "show", 100, 0, 1.0, &class_p);
//  status = cv::imwrite(class_img_p_name, class_p);
//  if (status) LOG(INFO) << class_img_p_name << "success.";

  // mask
//  std::string mask_img_name
//      = output_file_path_ + "mask" + Num2Str(frm_idx) + ".png";
//  status = cv::imwrite(mask_img_name, cam_set_[frm_idx].mask);
//  if (status) LOG(INFO) << class_img_name << " success.";

//  // pointer
//  std::string pointer_txt_name
//      = output_file_path_ + "pointer" + Num2Str(frm_idx) + ".txt";
//  if (frm_idx > 0) {
//    status = SaveImgMatToTxt(pointer_txt_name, cam_set_[frm_idx].pointer);
//  }

  // vertex_set
//  std::string vertex_txt_name
//      = output_file_path_ + "vertex_val" + Num2Str(frm_idx) + ".txt";
//  status = SaveValToTxt(vertex_txt_name, vertex_set_[frm_idx].vertex_val_,
//                        vertex_set_[frm_idx].block_height_,
//                        vertex_set_[frm_idx].block_width_);
//  if (status) LOG(INFO) << vertex_txt_name << " success.";
//  // vertex_valid
//  std::string valid_txt_name
//      = output_file_path_ + "valid" + Num2Str(frm_idx) + ".txt";
//  status = SaveVecUcharToTxt(valid_txt_name, vertex_set_[frm_idx].valid_,
//                             vertex_set_[frm_idx].block_height_,
//                             vertex_set_[frm_idx].block_width_);
//  if (status) LOG(INFO) << valid_txt_name << " success.";

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
//  cam_set_[frm_idx].img_class_p.release();
//  cam_set_[frm_idx].shade_mat.release();
//  cam_set_[frm_idx].img_est.release();
  cam_set_[frm_idx].x_pro.release();
  cam_set_[frm_idx].y_pro.release();
  cam_set_[frm_idx].depth.release();
  cam_set_[frm_idx].mask.release();
//  cam_set_[frm_idx].pointer.resize(0, 0);
//  cam_set_[frm_idx].km_center.resize(kIntensityClassNum, 0);
//  cam_set_[frm_idx].norm_vec.resize(3, 0);
  // Release NodeSet
  node_set_[frm_idx].Clear();
  // Release vertex_set
//  vertex_set_[frm_idx].Clear();
  LOG(INFO) << "End: ReleaseSpace(" << frm_idx << ")";
}

bool Reconstructor::Close() {
  return true;
}