//
// Created by pointer on 17-12-16.
//

#include "global_fun.h"

std::string Num2Str(int number) {
  std::stringstream ss;
  ss << number;
  std::string idx2str;
  ss >> idx2str;
  return idx2str;
}

void ErrorThrow(std::string error_info) {
  std::cout << "<Error>" << error_info << std::endl;
  fgetc(stdin);
}

cv::Mat LoadTxtToMat(std::string file_name, int kHeight, int kWidth) {
  cv::Mat result(kHeight, kWidth, CV_64FC1);
  std::fstream file(file_name, std::ios::in);
  if (!file) {
    ErrorThrow("LoadTxtToMat, file_name=" + file_name);
  }
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      double tmp;
      file >> tmp;
      result.at<double>(h, w) = tmp;
    }
  }
  file.close();
  return result;
}

bool SaveMatToTxt(std::string file_name, cv::Mat mat) {
  cv::Size mat_size = mat.size();
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    ErrorThrow("SaveMatToTxt, file_name=" + file_name);
    return false;
  }
  for (int h = 0; h < mat_size.height; h++) {
    for (int w = 0; w < mat_size.width; w++) {
      file << mat.at<double>(h, w) << " ";
    }
    file << "\n";
  }
  return true;
}

bool SaveVecToTxt(std::string file_name,
                  Eigen::Matrix<double, Eigen::Dynamic, 1> vec,
                  int kHeight, int kWidth) {
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    ErrorThrow("SaveVecToTxt, file_name=" + file_name);
    return false;
  }
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      file << vec(h * kWidth + w) << " ";
    }
    file << "\n";
  }
  return true;
}

bool SaveVecUcharToTxt(std::string file_name,
                       Eigen::Matrix<uchar, Eigen::Dynamic, 1> vec,
                       int kHeight, int kWidth) {
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    ErrorThrow("SaveVecToTxt, file_name=" + file_name);
    return false;
  }
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      file << vec(h * kWidth + w) << " ";
    }
    file << "\n";
  }
  return true;
}


int FloodFill(cv::Mat map, int h, int w, uchar replace, uchar valid_val) {
  int total_num = 0;
  cv::Size map_size = map.size();
  std::queue<cv::Point2i> my_queue;
  if (map.at<uchar>(h, w) == valid_val) {
    map.at<uchar>(h, w) = replace;
    total_num++;
    my_queue.push(cv::Point2i(w, h));
  }
  while (!my_queue.empty()) {
    cv::Point2i center_point = my_queue.front();
    my_queue.pop();
    int h_cen = center_point.y;
    int w_cen = center_point.x;
    // Up,Right,Down,Left
    if ((h_cen - 1 >= 0)
        && (map.at<uchar>(h_cen - 1, w_cen) == valid_val)) {
      map.at<uchar>(h_cen - 1, w_cen) = replace;
      total_num++;
      my_queue.push(cv::Point2i(w_cen, h_cen - 1));
    }
    if ((w_cen + 1 < map_size.width)
        && (map.at<uchar>(h_cen, w_cen + 1) == valid_val)) {
      map.at<uchar>(h_cen, w_cen + 1) = replace;
      total_num++;
      my_queue.push(cv::Point2i(w_cen + 1, h_cen));
    }
    if ((h_cen + 1 < map_size.height)
        && (map.at<uchar>(h_cen + 1, w_cen) == valid_val)) {
      map.at<uchar>(h_cen + 1, w_cen) = replace;
      total_num++;
      my_queue.push(cv::Point2i(w_cen, h_cen + 1));
    }
    if ((w_cen - 1 >= 0)
        && (map.at<uchar>(h_cen, w_cen - 1) == valid_val)) {
      map.at<uchar>(h_cen, w_cen - 1) = replace;
      total_num++;
      my_queue.push(cv::Point2i(w_cen - 1, h_cen));
    }
  }
  return total_num;
}

