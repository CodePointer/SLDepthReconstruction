//
// Created by pointer on 18-3-29.
// Include 2 parts:
//   DepthSegment & IntensitySlot.
// For every pixel in camera:
//   Include IntensitySlot[<class_num>].
//     IntensitySlot = DepthSegment[<??>].
//   Also, for IntensitySlot, a pointer used for opt is maintained.
//   The pointer was continous, while depth is not.
//

#ifndef SLDEPTHRECONSTRUCTION_INTENSITYSLOT_H
#define SLDEPTHRECONSTRUCTION_INTENSITYSLOT_H

#include <static_para.h>
#include <global_fun.h>

struct DepthSegment {
  double start = -1;
  double end = -1;
  void Clear() {
    start = -1;
    end = -1;
  }
};

class IntensitySlot {
private:
  std::vector<DepthSegment> seg_sets_;
  std::vector<double> p_bias_;
  std::string info_;

public:
  IntensitySlot();
  ~IntensitySlot();
  bool InsertSegment(DepthSegment seg);
  double GetNearestPointerFromDepth(double depth);
  //double GetDepthFromPointer(double p);
  double GetPointerFromDepth(double depth);
  template <class T>
  T GetDepthFromPointer(T p) {
    if (seg_sets_.size() <= 0) {
      LOG(ERROR) << "Empty seg sets." << std::endl;
    }
    if (p < T(0)) {
      return T(seg_sets_[0].start);
    }
    for (int i = 0; i < p_bias_.size() - 1; i++) {
      T p_b = T(p_bias_[i + 1]);
      if (T(p_bias_[i + 1]) >= p) {
        return p + T(seg_sets_[i].start) - T(p_bias_[i]);
      }
    }
    return T(seg_sets_[seg_sets_.size() - 1].end);
  }
  std::string OutputSlot() {
    std::string result = "[";
    for (int i = 0; i < seg_sets_.size(); i++) {
      result += "[" + Val2Str(seg_sets_[i].start);
      result += "," + Val2Str(seg_sets_[i].end) + "]";
      if (i < seg_sets_.size() - 1) {
        result += ",";
      }
    }
    result += "]";
    return result;
  }
  std::string OutputBias() {
    std::string result = "[";
    for (int i = 0; i < p_bias_.size(); i++) {
      result += Val2Str(p_bias_[i]);
      if (i < p_bias_.size() - 1) {
        result += ",";
      }
    }
    result += "]";
    return result;
  }
  bool WriteToFile(std::string file_name) {
    std::string seg_file_name = file_name + "_s.txt";
    std::fstream file(seg_file_name, std::ios::out);
    for (int i = 0; i < seg_sets_.size(); i++) {
      file << seg_sets_[i].start << " " << seg_sets_[i].end << std::endl;
    }
    file.close();

    std::string p_file_name = file_name + "_p.txt";
    std::fstream pfile(p_file_name, std::ios::out);
    for (int i = 0; i < p_bias_.size(); i++) {
      pfile << p_bias_[i] << std::endl;
    }
    pfile.close();
  }
};

class CamSlotsMat {
public:
  IntensitySlot *** slots_;
  CamSlotsMat();
  ~CamSlotsMat();
};


#endif //SLDEPTHRECONSTRUCTION_INTENSITYSLOT_H
