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

std::string Num2Str(int number);

void ErrorThrow(std::string error_info);

cv::Mat LoadTxtToMat(std::string file_name, int kHeight, int kWidth);

bool SaveMatToTxt(std::string file_name, cv::Mat mat);

bool SaveValToTxt(std::string file_name,
                  Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> vec,
                  int kHeight, int kWidth);

bool SaveVecUcharToTxt(std::string file_name,
                       Eigen::Matrix<uchar, Eigen::Dynamic, 1> vec,
                       int kHeight, int kWidth);

int FloodFill(cv::Mat map, int h, int w, uchar replace, uchar valid_val);

#endif //SLDEPTHRECONSTRUCTION_GLOBAL_FUN_H
