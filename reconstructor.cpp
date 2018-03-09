//
// Created by pointer on 17-12-15.
//


#include "reconstructor.h"

Reconstructor::Reconstructor() {
  cam_set_ = nullptr;
  vertex_set_ = nullptr;
  pat_grid_ = nullptr;
}

Reconstructor::~Reconstructor() {
  if (cam_set_ != nullptr) {
    delete[]cam_set_;
    cam_set_ = nullptr;
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
  // Set file path
  main_file_path_ = "/home/pointer/CLionProjects/Data/20180122/HandMove/";
  pattern_file_name_ = "pattern_gauss";
  pattern_file_suffix_ = ".txt";
  dyna_file_path_ = "cam_0/dyna/";
  dyna_file_name_ = "dyna_mat";
  dyna_file_suffix_ = ".png";
  pro_file_path_ = "cam_0/pro/";
  pro_file_name_ = "xpro_mat";
  pro_file_suffix_ = ".txt";
  epi_A_file_name_ = "cam0_pro/EpiLine_A.txt";
  epi_B_file_name_ = "cam0_pro/EpiLine_B.txt";
  hard_mask_file_name_ = "cam0_pro/hard_mask.png";
  // Set output file path
  output_file_path_ = "/home/pointer/CLionProjects/Data/20180122/HandMove/result/";
  depth_file_path_ = "";
  depth_file_name_ = "depth";
  vertex_file_path_ = "";
  vertex_file_name_ = "vertex";
  valid_file_path_ = "";
  valid_file_name_ = "valid_vex";

  // Load Informations
  status = LoadDatasFromFiles();
  // Set pattern grid

//  cv::namedWindow("pattern");
//  cv::imshow("pattern", pattern_ / 255.0);
//  cv::waitKey(0);
//  cv::destroyWindow("pattern");
  return status;
}

bool Reconstructor::LoadDatasFromFiles() {
  // Calib_Set:M, D, cam_0, cam_1
  calib_set_.cam << 2426.231977104875, 0, 634.7636169096244,
      0, 2423.019176341463, 422.664603232326,
      0, 0, 1;
  calib_set_.pro << 1910.445054230685, 0, 674.7918402916996,
      0, 1919.469009195578, 670.8165424942966,
      0, 0, 1;
  calib_set_.R << 0.9641898953391403, -0.009497729569831175, 0.2650427113861386,
      0.0486242666266381, 0.9887495098811747, -0.1414570161029787,
      -0.2607173304959209, 0.1492789330172048, 0.9538041065839074;
  calib_set_.t << -6.309369886992315,
      -1.716176874067702,
      5.309834884842104;
//  calib_set_.light_vec_ << -0.12036800,
//      0.13673927,
//      -0.98326696;
  calib_set_.light_vec_ << 0.13673927,
      -0.12036800,
      -0.98326696;

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
  pattern_ = LoadTxtToMat(main_file_path_ + pattern_file_name_
                                + pattern_file_suffix_,
                                kProHeight, kProWidth);
  ceres::Grid2D<double, 1> * pat_grid = new ceres::Grid2D<double, 1>(
      (double*)pattern_.data, 0, kProHeight, 0, kProWidth);
  pat_grid_ = new ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>>(*pat_grid);

  // EpiLine set
  epi_A_mat_ = LoadTxtToMat(main_file_path_ + epi_A_file_name_,
                                  kCamHeight, kCamWidth);
  epi_B_mat_ = LoadTxtToMat(main_file_path_ + epi_B_file_name_,
                                  kCamHeight, kCamWidth);
  hard_mask_ = cv::imread(main_file_path_ + hard_mask_file_name_, cv::IMREAD_GRAYSCALE);
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (hard_mask_.at<uchar>(h, w) == 255) {
        hard_mask_.at<uchar>(h, w) = my::VERIFIED_TRUE;
      } else {
        hard_mask_.at<uchar>(h, w) = my::VERIFIED_FALSE;
      }
    }
  }
  // Cam_set
  cam_set_ = new CamMatSet[kFrameNum];
  vertex_set_ = new VertexSet[kFrameNum];
  for (int frm_idx = 0; frm_idx < kFrameNum; frm_idx++) {
    cam_set_[frm_idx].img_obs = cv::imread(main_file_path_
                                           + dyna_file_path_
                                           + dyna_file_name_
                                           + Num2Str(frm_idx)
                                           + dyna_file_suffix_,
                                           cv::IMREAD_GRAYSCALE);
//    cam_set_[frm_idx].x_pro = LoadTxtToMat(main_file_path_
//                                           + pro_file_path_
//                                           + pro_file_name_
//                                           + Num2Str(frm_idx)
//                                           + pro_file_suffix_,
//                                           kCamHeight, kCamWidth);
//    cam_set_[frm_idx].y_pro = LoadTxtToMat(main_file_path_
//                                           + pro_file_path_
//                                           + "ypro_mat"
//                                           + Num2Str(frm_idx)
//                                           + pro_file_suffix_,
//                                           kCamHeight, kCamWidth);
//    printf("1\n");
//    SetMaskMatFromXpro(frm_idx);
//    ConvXpro2Depth(&cam_set_[frm_idx]);
//    GenerateIest(frm_idx);
//    cam_set_[frm_idx].img_est.copyTo(cam_set_[frm_idx].img_obs);
//    cam_set_[frm_idx].img_est.setTo(0);
//    cv::imwrite(main_file_path_ + dyna_file_path_ + dyna_file_name_
//                + Num2Str(frm_idx) + dyna_file_suffix_,
//                cam_set_[frm_idx].img_obs);
  }
  // Load first data: x_pro & depth
  SetMaskMatFromIobs(0);
  cam_set_[0].x_pro = LoadTxtToMat(main_file_path_
                                       + pro_file_path_
                                       + pro_file_name_
                                       + Num2Str(0)
                                       + pro_file_suffix_,
                                       kCamHeight, kCamWidth);
  ConvXpro2Depth(&cam_set_[0]);
  SetVertexFromBefore(0);
  return true;
}

void Reconstructor::ConvXpro2Depth(CamMatSet *ptr_cam_set) {
  ptr_cam_set->depth.create(kCamHeight, kCamWidth, CV_64FC1);
  Eigen::Vector3d vec_D = calib_set_.D;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (ptr_cam_set->mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      int idx_k = h * kCamWidth + w;
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
      Eigen::Vector3d vec_M = calib_set_.M.block<3, 1>(0, idx_k);
      double depth = - (vec_D(0) - vec_D(2) * x_pro)
                     / (vec_M(0) - vec_M(2) * x_pro);
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
}

bool Reconstructor::Run() {
  bool status = true;
  CalculateDepthMat(0);
  GenerateIest(0);

  std::string I_est_name = output_file_path_ + "I_est" + Num2Str(0) + ".png";
  cv::imwrite(I_est_name, cam_set_[0].img_est);

  for (int frm_idx = 1; (frm_idx < kFrameNum) && status; frm_idx++) {
    std::cout << "Frame " << frm_idx << ":" << std::endl;
    // Set a mask according to I_obs
    SetMaskMatFromIobs(frm_idx);

    // I_obs -> binary
//    cv::Mat tmp_obs;
//    int blur_size = 11;
//    cam_set_[frm_idx].img_obs.copyTo(tmp_obs);
//    for (int h = 0; h < kCamHeight; h++) {
//      for (int w = 0; w < kCamWidth; w++) {
//        if (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
//          double sum_val = 0;
//          int sum_num = 0;
//          for (int dh = -blur_size; dh <= blur_size; dh++) {
//            for (int dw = -blur_size; dw <= blur_size; dw++) {
//              int h_n = h + dh;
//              int w_n = w + dw;
//              if ((h_n < 0) || (h_n >= kCamHeight) || (w_n < 0) || (w_n >= kCamWidth))
//                continue;
//              if (cam_set_[frm_idx].mask.at<uchar>(h_n, w_n) == my::VERIFIED_TRUE) {
//                sum_num++;
//                sum_val += double(tmp_obs.at<uchar>(h_n, w_n));
//              }
//            }
//          }
//          double val = double(cam_set_[frm_idx].img_obs.at<uchar>(h, w));
//          double avr_val = sum_val / sum_num;
//          if (val > avr_val) {
//            cam_set_[frm_idx].img_obs.at<uchar>(h, w) = 1;
//          } else {
//            cam_set_[frm_idx].img_obs.at<uchar>(h, w) = 0;
//          }
//        }
//      }
//    }
//    cv::namedWindow("I_obs" + Num2Str(frm_idx));
//    cv::imshow("I_obs" + Num2Str(frm_idx), cam_set_[frm_idx].img_obs);
//    cv::waitKey(200);
//    cv::destroyWindow("I_obs" + Num2Str(frm_idx));

    // Set vertex initial value
    SetVertexFromBefore(frm_idx);
    CalculateDepthMat(frm_idx);
//    SaveValToTxt("vertex_before.txt", vertex_set_[frm_idx].vertex_val_,
//                 vertex_set_[frm_idx].block_height_,
//                 vertex_set_[frm_idx].block_width_);
    // Optimization
    if (status)
      status = OptimizeVertexSet(frm_idx);
//    SaveValToTxt("vertex_after.txt", vertex_set_[frm_idx].vertex_val_,
//                 vertex_set_[frm_idx].block_height_,
//                 vertex_set_[frm_idx].block_width_);
    // Build up depth mat from mask
    if (status)
      status = CalculateDepthMat(frm_idx);
    // Calculate Estimated Image from depth mat
    if (status)
      status = GenerateIest(frm_idx);
    // Write result to file
    if (status)
      status = WriteResult(frm_idx);
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

void Reconstructor::SetMaskMatFromIobs(int frm_idx) {
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

  // Normalize obs image
  uchar max_val = 0;
  uchar min_val = 255;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
        if (cam_set_[frm_idx].img_obs.at<uchar>(h, w) > max_val)
          max_val = cam_set_[frm_idx].img_obs.at<uchar>(h, w);
        if (cam_set_[frm_idx].img_obs.at<uchar>(h, w) < min_val)
          min_val = cam_set_[frm_idx].img_obs.at<uchar>(h, w);
      } else {
        cam_set_[frm_idx].img_obs.at<uchar>(h, w) = 0;
      }
    }
  }
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::VERIFIED_TRUE) {
        double ori_val = cam_set_[frm_idx].img_obs.at<uchar>(h, w);
        double new_val = (ori_val - min_val) / (double)(max_val - min_val);
        cam_set_[frm_idx].img_obs.at<uchar>(h, w) = (uchar)(new_val * 255);
      }
    }
  }
  cv::namedWindow("I_obs" + Num2Str(frm_idx));
  cv::imshow("I_obs" + Num2Str(frm_idx), cam_set_[frm_idx].img_obs);
  cv::imwrite("I_obs" + Num2Str(frm_idx) + ".png", cam_set_[frm_idx].img_obs);
  cv::waitKey(2000);
  cv::destroyWindow("I_obs" + Num2Str(frm_idx));
//  cv::namedWindow("mask" + Num2Str(frm_idx));
//  cv::imshow("mask" + Num2Str(frm_idx), cam_set_[frm_idx].mask * 255);
//  cv::waitKey(00);
//  cv::destroyWindow("mask" + Num2Str(frm_idx));
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
    // Check last frame: last frame have value, Set to VERIFIED_TRUE.
    // Otherwise, set to INITIAL_TRUE.
    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) {
      int x = vertex_set_[frm_idx].pos_(i, 0);
      int y = vertex_set_[frm_idx].pos_(i, 1);
      if (frm_idx == 3 && x == 735 && y == 825) {
//        std::cout << "Found." << std::endl;
      }
      if (cam_set_[frm_idx].mask.at<uchar>(y, x) == my::VERIFIED_TRUE) {
        if (frm_idx == 3 && x == 735 && y == 825) {
//          std::cout << "VERIFIED_TRUE." << std::endl;
        }
        uchar set_flag = my::VERIFIED_TRUE;
        // 1. Last frame, vertex is not valid, set INITIAL
        if (vertex_set_[frm_idx - 1].valid_(i) != my::VERIFIED_TRUE)
          set_flag = my::INITIAL_TRUE;
        if (vertex_set_[frm_idx - 1].vertex_val_(i, 0) < 0)
          set_flag = my::INITIAL_TRUE;
        // 2. Last frame, mask is false, set to INITIAL
        if (cam_set_[frm_idx - 1].mask.at<uchar>(y, x) != my::VERIFIED_TRUE)
          set_flag = my::INITIAL_TRUE;
        // 3. Last frame, exceeded gradient, set to INITIAL
//        double sum_val = 0;
//        int sum_num = 0;
//        int idx_up = vertex_set_[frm_idx - 1].GetVertexIdxByPos(x, y - kGridSize);
//        if (idx_up > 0) {
//          sum_val += vertex_set_[frm_idx - 1].vertex_val_(i)
//                     - vertex_set_[frm_idx - 1].vertex_val_(idx_up);
//          sum_num += 1;
//        }
//        int idx_dn = vertex_set_[frm_idx - 1].GetVertexIdxByPos(x, y + kGridSize);
//        if (idx_dn > 0) {
//          sum_val += vertex_set_[frm_idx - 1].vertex_val_(i)
//                     - vertex_set_[frm_idx - 1].vertex_val_(idx_dn);
//          sum_num += 1;
//        }
//        int idx_lf = vertex_set_[frm_idx - 1].GetVertexIdxByPos(x - kGridSize, y);
//        if (idx_lf > 0) {
//          sum_val += vertex_set_[frm_idx - 1].vertex_val_(i)
//                     - vertex_set_[frm_idx - 1].vertex_val_(idx_lf);
//          sum_num += 1;
//        }
//        int idx_rt = vertex_set_[frm_idx - 1].GetVertexIdxByPos(x + kGridSize, y);
//        if (idx_rt > 0) {
//          sum_val += vertex_set_[frm_idx - 1].vertex_val_(i)
//                     - vertex_set_[frm_idx - 1].vertex_val_(idx_rt);
//          sum_num += 1;
//        }
//        double grad = sum_val * 4 / sum_num;
//        if (grad > 8.0)
//          set_flag = my::INITIAL_TRUE;

        vertex_set_[frm_idx].valid_(i) = set_flag;
        if (set_flag == my::VERIFIED_TRUE) {
          vertex_set_[frm_idx].vertex_val_.block(i, 0, 1, 4)
              = vertex_set_[frm_idx - 1].vertex_val_.block(i, 0, 1, 4);
        }
      } else {
        vertex_set_[frm_idx].vertex_val_(i, 0) = -1;
        vertex_set_[frm_idx].valid_(i) = my::VERIFIED_FALSE;
        if (frm_idx == 3 && x == 735 && y == 825) {
//          std::cout << "VERIFIED_FALSE. " << i << std::endl;
//          std::cout << vertex_set_[frm_idx].vertex_val_(i) << std::endl;
//          std::cout << (int)vertex_set_[frm_idx].valid_(i) << std::endl;
        }
      }
    }
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
    for (int i = 0; i < vertex_set_[frm_idx].len_; i++) { // Check
      if ((vertex_set_[frm_idx].valid_(i) == my::VERIFIED_TRUE)
          && (vertex_set_[frm_idx].vertex_val_(i, 0) <= 0)) {
        ErrorThrow("Vertex value less than 0. i=" + Num2Str(i));
      }
    }

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

  // Add blocks
  double fx = calib_set_.cam_mat(0, 0);
  double fy = calib_set_.cam_mat(1, 1);
  double dx = calib_set_.cam_mat(0, 2);
  double dy = calib_set_.cam_mat(1, 2);
  int block_num = 0;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if ((cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)){
        continue;
      }
      // Add Data Term
      Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
          = vertex_set_[frm_idx].FindkNearestVertex(w, h, kNearestPoints + 1);
      double* data_set = vertex_set_[frm_idx].vertex_val_.data();
      std::vector<double*> parameter_blocks;
      int idx_k = h*kCamWidth + w;
      double img_obs = (double)(cam_set_[frm_idx].img_obs.at<uchar>(h, w));
      Eigen::Matrix<double, 3, 1> vec_M = calib_set_.M.block<3, 1>(0, idx_k);
      Eigen::Matrix<double, 3, 1> vec_D = calib_set_.D;
      double epi_A = epi_A_mat_.at<double>(h, w);
      double epi_B = epi_B_mat_.at<double>(h, w);
      DeformConstraint::DeformCostFunction* cost_function =
          DeformConstraint::Create(
              *pat_grid_, img_obs, kNearestPoints, vec_M, vec_D,
              epi_A, epi_B, fx, fy, dx, dy,
              calib_set_.light_vec_, vertex_nbr, data_set, &parameter_blocks);
      problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);
      block_num++;
    }
  }

  // Add residual block: Regular Term
  for (int ver_idx = 0; ver_idx < vertex_set_[frm_idx].len_; ver_idx++) {
    int w = vertex_set_[frm_idx].pos_(ver_idx, 0);
    int h = vertex_set_[frm_idx].pos_(ver_idx, 1);
    if (vertex_set_[frm_idx].valid_(ver_idx) != my::VERIFIED_TRUE)
      continue;
    Eigen::Matrix<double, Eigen::Dynamic, 2> vertex_nbr
        = vertex_set_[frm_idx].FindkNearestVertex(w, h, kRegularNbr + 2);
    double* data_set = vertex_set_[frm_idx].vertex_val_.data();
    std::vector<double*> parameter_blocks;
    RegularConstraint::RegularCostFunction* cost_function =
        RegularConstraint::Create(
            alpha_val, alpha_norm, kRegularNbr, vertex_nbr, data_set,
            &parameter_blocks);
    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);
    block_num++;
  }

  ceres::Solver::Options options;
  options.gradient_tolerance = 1e-10;
  options.function_tolerance = 1e-10;
  options.min_relative_decrease = 1e-1;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 50;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;

//  if (frm_idx == 3) {
//    int idx = vertex_set_[frm_idx].GetVertexIdxByPos(735, 825);
//    std::cout << "Before Opt: " << idx << std::endl;
//    std::cout << vertex_set_[frm_idx].vertex_val_(idx) << std::endl;
//    std::cout << (int)vertex_set_[frm_idx].valid_(idx) << std::endl;
//  }

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
//      cam_set_[frm_idx].depth.at<double>(h, w) = d_k;
      if ((d_k >= kDepthMin) && (d_k <= kDepthMax)) {
        cam_set_[frm_idx].depth.at<double>(h, w) = d_k;
      } else {
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::MARKED;
      }
      if (h == 885 && w == 705) {
//        std::cout << nb_idx_set << std::endl;
//        printf("d_k = %f\n", d_k);
//        printf("vertex_val_ul[%d] = %f\n", ul_idx, v_ul);
      }
    }
  }
  // Interpolation
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) == my::MARKED) {
        int search_rad = 5;
        double sum_val = 0;
        int sum_num = 0;
        for (int d_h = -search_rad; d_h <= search_rad; d_h++) {
          for (int d_w = -search_rad; d_w <= search_rad; d_w++) {
            if (cam_set_[frm_idx].mask.at<uchar>(h+d_h, w+d_w) == my::VERIFIED_TRUE) {
              sum_num++;
              sum_val += cam_set_[frm_idx].depth.at<double>(h+d_h, w+d_w);
            }
          }
        }
        cam_set_[frm_idx].depth.at<double>(h, w) = sum_val / double(sum_num);
        cam_set_[frm_idx].mask.at<uchar>(h, w) = my::VERIFIED_TRUE;
      }
    }
  }
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
  cv::Mat norm_mat;
  norm_mat.create(kCamHeight, kCamWidth, CV_8UC1);
  norm_mat.setTo(0);
  Eigen::Vector3d D = calib_set_.D;
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      if (cam_set_[frm_idx].mask.at<uchar>(h, w) != my::VERIFIED_TRUE)
        continue;
      // Get depth
      double depth = cam_set_[frm_idx].depth.at<double>(h, w);
      if (depth <= 0)
        continue;
      int idx_k = h*kCamWidth + w;
      Eigen::Vector3d M = calib_set_.M.block<3, 1>(0, idx_k);
      double epi_A = epi_A_mat_.at<double>(h, w);
      double epi_B = epi_B_mat_.at<double>(h, w);
      double x_pro = (M(0)*depth + D(0)) / (M(2)*depth + D(2));
      double y_pro = (-epi_A/epi_B)*x_pro + 1 / epi_B;
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

bool Reconstructor::Close() {
  return true;
}