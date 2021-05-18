/* Copyright 2021 The Kingsoft AI Toolkits Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"

namespace KSAICommonOP {
  void LimitMaxSide(const cv::Mat &img_in, cv::Mat &img_out,
                    const float &max_side_len);
  void ResizeImageWithMultiple32(const cv::Mat &img_in, cv::Mat &img_out);
  void ResizeImageKeepRatio(const cv::Mat &img_in,
                            cv::Mat &img_out, const int &fixed_height);
  void PaddingImage(const cv::Mat &img_in, cv::Mat &img_out, 
                    const int &top, const int &bottom,
                    const int &left, const int &right);
  bool NeedInverseColor(const cv::Mat &img);
};