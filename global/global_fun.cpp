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

std::string Val2Str(double val) {
  std::stringstream ss;
  ss << val;
  std::string idx2str;
  ss >> idx2str;
  return idx2str;
}

double GetDepthFromXpro(double x_pro, int h_cam, int w_cam, CalibSet* p_calib) {
  int idx = h_cam * kCamWidth + w_cam;
  Eigen::Vector3d vec_M = p_calib->M.block<3, 1>(0, idx);
  Eigen::Vector3d vec_D = p_calib->D;
  double depth = - (vec_D(0) - vec_D(2) * x_pro)
                 / (vec_M(0) - vec_M(2) * x_pro);
  return depth;
}

double GetXproFromDepth(double depth, int h_cam, int w_cam, CalibSet* p_calib) {
  int idx = h_cam * kCamWidth + w_cam;
  Eigen::Vector3d vec_M = p_calib->M.block<3, 1>(0, idx);
  Eigen::Vector3d vec_D = p_calib->D;
  double x_pro = (vec_M(0)*depth + vec_D(0))
                 / (vec_M(2)*depth + vec_D(2));
  return x_pro;
}

double GetYproFromXpro(double x_pro, int h_cam, int w_cam, cv::Mat epi_A, cv::Mat epi_B) {
  double A = epi_A.at<double>(h_cam, w_cam);
  double B = epi_B.at<double>(h_cam, w_cam);
  double y_pro = (-A/B)*x_pro + 1/B;
  return y_pro;
}

// Used for weight calculation
void CalculateNbrWeight(Eigen::Matrix<double, Eigen::Dynamic, 2> nbr_set,
                        Eigen::Matrix<double, Eigen::Dynamic, 1> * weight,
                        int k, int start_idx) {
  if (weight == nullptr) {
    return;
  }
  double sum_val = 0;
  for (int i = start_idx; i < start_idx + k; i++) {
    (*weight)(i) = pow((1 - nbr_set(i, 1) / nbr_set(start_idx + k, 1)), 2);
    sum_val += (*weight)(i);
  }
  (*weight) = (*weight) / sum_val;
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

void LoadTxtToEigen(std::string file_name, int kHeight, int kWidth, double * data) {
  std::fstream file(file_name, std::ios::in);
  if (!file) {
    ErrorThrow("LoadTxtToEigen, file_name=" + file_name);
  }
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      double tmp;
      file >> tmp;
//      data[w*kHeight + h] = tmp;
      data[h*kWidth + w] = tmp;
    }
  }
  file.close();
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
  file.close();
  return true;
}

bool SaveImgMatToTxt(std::string file_name, ImgMatrix mat) {
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    LOG(ERROR) << "file_name=" << file_name;
    return false;
  }
  for (int h = 0; h < kCamHeight; h++) {
    for (int w = 0; w < kCamWidth; w++) {
      file << mat(h, w) << " ";
    }
    file << "\n";
  }
  file.close();
  return true;
}

bool SaveValToTxt(std::string file_name,
                  Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor> vec,
                  int kHeight, int kWidth) {
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    ErrorThrow("SaveValToTxt, file_name=" + file_name);
    return false;
  }
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      file << vec(h * kWidth + w, 0) << " ";
      file << vec(h * kWidth + w, 1) << " ";
      file << vec(h * kWidth + w, 2) << " ";
      file << vec(h * kWidth + w, 3) << " ";
      file << "\n";
    }
//    file << "\n";
  }
  file.close();
  return true;
}

bool SaveFrmToTxt(std::string file_name,
                  Eigen::Matrix<int, Eigen::Dynamic, 1> vec,
                  int kHeight, int kWidth) {
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    ErrorThrow("SaveFrmToTxt, file_name=" + file_name);
    return false;
  }
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      file << vec(h * kWidth + w, 0);
      file << "\n";
    }
  }
  file.close();
  return true;
}

bool SaveVecUcharToTxt(std::string file_name,
                       Eigen::Matrix<uchar, Eigen::Dynamic, 1> vec,
                       int kHeight, int kWidth) {
  std::fstream file(file_name, std::ios::out);
  if (!file) {
    ErrorThrow("SaveValToTxt, file_name=" + file_name);
    return false;
  }
  for (int h = 0; h < kHeight; h++) {
    for (int w = 0; w < kWidth; w++) {
      file << vec(h * kWidth + w) << " ";
    }
    file << "\n";
  }
  file.close();
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

