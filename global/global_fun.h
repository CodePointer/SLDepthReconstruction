//
// Created by pointer on 17-12-15.
//

#ifndef SLDEPTHRECONSTRUCTION_GLOBAL_FUN_H
#define SLDEPTHRECONSTRUCTION_GLOBAL_FUN_H

#include <sstream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <queue>
#include <Eigen/Eigen>
#include "static_para.h"

std::string Num2Str(int number);

std::string Val2Str(double val);

double GetDepthFromXpro(double x_pro, int x_cam, int y_cam, CalibSet* p_calib);

double GetXproFromDepth(double depth, int h_cam, int w_cam, CalibSet* p_calib);

double GetYproFromXpro(double x_pro, int x_cam, int y_cam, cv::Mat epi_A, cv::Mat epi_B);

// Used for weight calculation
void CalculateNbrWeight(Eigen::Matrix<double, Eigen::Dynamic, 2> nbr_set,
                        Eigen::Matrix<double, Eigen::Dynamic, 1> * weight,
                        int k, int start_idx = 0);

void ErrorThrow(std::string error_info);

cv::Mat LoadTxtToMat(std::string file_name, int kHeight, int kWidth);

// Eigen::rowMajor; txt is rowMajor
void LoadTxtToEigen(std::string file_name,
                    int kHeight, int kWidth, double * data);

bool SaveMatToTxt(std::string file_name, cv::Mat mat);

bool SaveImgMatToTxt(std::string file_name, ImgMatrix mat);

bool SaveValToTxt(std::string file_name,
                  Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> vec,
                  int kHeight, int kWidth);

bool SaveFrmToTxt(std::string file_name,
                  Eigen::Matrix<int, Eigen::Dynamic, 1> vec,
                  int kHeight, int kWidth);

bool SaveVecUcharToTxt(std::string file_name,
                       Eigen::Matrix<uchar, Eigen::Dynamic, 1> vec,
                       int kHeight, int kWidth);

int FloodFill(cv::Mat map, int h, int w, uchar replace, uchar valid_val);

template <class T>
int ShowMat(cv::Mat * mat, std::string win_name, int delay, T min, T max,
            bool close_flag = true,
            cv::Mat * p_norm = nullptr) {
  cv::Mat show_mat;
  show_mat.create(mat->size(), CV_8UC1);
  if (max <= min) {
    LOG(ERROR) << "max <= min (" << max << ", " << min << ")";
    return -1;
  }
  for (int h = 0; h < mat->size().height; h++) {
    for (int w = 0; w < mat->size().width; w++) {
      if (mat->at<T>(h, w) <= min) {
        show_mat.at<uchar>(h, w) = 0;
        continue;
      }
      if (mat->at<T>(h, w) >= max) {
        show_mat.at<uchar>(h, w) = 255;
        continue;
      }
      show_mat.at<uchar>(h, w) = (uchar)(255 * (mat->at<T>(h, w) - min) / (max - min));
    }
  }
  if (p_norm != nullptr) {
    show_mat.copyTo(*p_norm);
  }

  if (close_flag) {
    cv::namedWindow(win_name);
  }
  cv::imshow(win_name, show_mat);
  int key = cv::waitKey(delay);
  if (close_flag) {
    cv::destroyWindow(win_name);
  }
  return key;
}

#endif //SLDEPTHRECONSTRUCTION_GLOBAL_FUN_H
