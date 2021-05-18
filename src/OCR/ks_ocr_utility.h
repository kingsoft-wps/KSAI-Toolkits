/* Copyright 2021 The kingsoft AI Toolkits Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
//
// Part of the following code in this file refs to
// https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/deploy/cpp_infer/src/postprocess_op.cpp
//
==============================================================================*/
#pragma once
#include <iostream>

#include "ks_data_type.h"
#include "opencv2/opencv.hpp"

namespace KSAIOCRUtility {
  bool CreateDict(const std::string &dict_path, std::map<int, std::string> &char_dict);
  void MergeHorizontal(VOCRectf &rects, std::vector<std::vector<int>> &labels,
                       std::vector<std::string> &results,
                       std::vector<std::vector<int>> &duanluo_results_labels,
                       std::vector<std::string> &merged_results);
  bool CTCGreedyDecoder(const std::vector<int> &shape, const float *input,
                        std::vector<std::vector<int>> &output,
                        std::vector<float> &logit);
  float BoxScoreFast(const std::vector<std::vector<float>> &box_array, const cv::Mat &pred);
  std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box,float& ssid, float& min_size);
  std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(
      const cv::Mat pred, const cv::Mat bitmap, const float &box_thresh, 
      const float &det_db_unclip_ratio, int dest_width, int dest_height);
  cv::RotatedRect UnClip(std::vector<std::vector<float>> box, const float &unclip_ratio);
  void GetContourArea(const std::vector<std::vector<float>> &box,
                      const float unclip_ratio, float &distance);
  static bool XsortInt(const std::vector<int> a, const std::vector<int> b);
  static bool XsortFp32(const std::vector<float> a, const std::vector<float> b);
  std::vector<std::vector<float>> Mat2Vector(const cv::Mat mat);
  void GetWarpImgs(const cv::Mat &img, const VOCRectf &rects, std::vector<cv::Mat> &warp_imgs);

  template <class T> inline T Clamp(T x, T min, T max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

  inline float Clampf(float x, float min, float max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }
};
