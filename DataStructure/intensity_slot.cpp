//
// Created by pointer on 18-3-29.
//

#include "intensity_slot.h"

IntensitySlot::IntensitySlot() {
  seg_sets_.clear();
  p_bias_.clear();
  p_bias_.push_back(0);
}

IntensitySlot::~IntensitySlot() {

}

bool IntensitySlot::InsertSegment(DepthSegment seg) {
  bool status = true;
  if (seg.end <= seg.start) {
    LOG(ERROR) << "Illegal segment.";
    return false;
  }
  seg_sets_.push_back(seg);
  int idx = (int)seg_sets_.size() - 1;
  p_bias_.push_back(p_bias_[idx] + seg.end - seg.start);
  return true;
}

// Give an initial depth return a fixed pointer.
double IntensitySlot::GetNearestPointerFromDepth(double depth) {
  double pointer = -1.0;
  if (seg_sets_.empty()) {
//    LOG(ERROR) << "Empty intensity slot.";
    return -1.0;
  }
  for (int i = 0; i < seg_sets_.size(); i++) {
    if (depth >= seg_sets_[i].start && depth <= seg_sets_[i].end) {
      pointer = GetPointerFromDepth(depth);
      break;
    } else if (depth < seg_sets_[i].start) {
      if (i == 0) {
        pointer = GetPointerFromDepth(seg_sets_[i].start);
        break;
      } else {
        pointer = (std::abs(depth - seg_sets_[i-1].end)
                   < std::abs(depth - seg_sets_[i].start))
                  ? GetPointerFromDepth(seg_sets_[i-1].end)
                  : GetPointerFromDepth(seg_sets_[i].start);
        break;
      }
    }
  }
  if (pointer < 0) {
    pointer = GetPointerFromDepth(seg_sets_[seg_sets_.size() - 1].end);
  }
  return pointer;
}

//double IntensitySlot::GetDepthFromPointer(double p) {
//  for (int i = 0; i < p_bias_.size(); i++) {
//    if (p_bias_[i + 1] >= p) {
//      return p + seg_sets_[i].start - p_bias_[i];
//    }
//  }
//}

double IntensitySlot::GetPointerFromDepth(double depth) {
  for (int i = 0; i < seg_sets_.size(); i++) {
    if (depth >= seg_sets_[i].start && depth <= seg_sets_[i].end) {
      return depth - seg_sets_[i].start + p_bias_[i];
    }
  }
}

//template <class T>
//T IntensitySlot::GetDepthFromPointer(T p) {
//  for (int i = 0; i < p_bias_.size(); i++) {
//    if (T(p_bias_[i + 1]) >= p) {
//      return p + T(seg_sets_[i].start) - T(p_bias_[i]);
//    }
//  }
//}

CamSlotsMat::CamSlotsMat() {
  slots_ = new IntensitySlot**[kCamHeight];
  for (int h = 0; h < kCamHeight; h++) {
    slots_[h] = new IntensitySlot*[kCamWidth];
    for (int w = 0; w < kCamWidth; w++) {
      slots_[h][w] = nullptr;
    }
  }
}

CamSlotsMat::~CamSlotsMat() {
  if (slots_ != nullptr) {
    for (int h = 0; h < kCamHeight; h++) {
      if (slots_[h] != nullptr) {
        for (int w = 0; w < kCamWidth; w++) {
          if (slots_[h][w] != nullptr) {
            delete[](slots_[h][w]);
          }
        }
      }
      delete[](slots_[h]);
    }
  }
  delete[]slots_;
}