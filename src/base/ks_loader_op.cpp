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
#include "ks_loader_op.h"

namespace KSAILoaderOp{

  cv::Mat LoadImageFromFile(const std::string &file_path) {
    if (file_path.empty()) {
      std::cout << "image path is invalid !!" << std::endl;
      return cv::Mat();
    }
    cv::Mat image = cv::imread(file_path);
    return image;
  }

  cv::Mat LoadImageFromBuffer(char *buffer, int width, int height, int channels) {
    if (!buffer) {
      return cv::Mat();
    }
    if (height <= 0 || width <= 0) {
      return cv::Mat();
    }
    if (channels != 4) {
      return cv::Mat();
    }
    cv::Mat input(height, width, CV_8UC4, buffer);
    if (input.empty()) {
      return cv::Mat();
    }
    if (!input.data) {
      return cv::Mat();
    }
    if (input.channels() != 4) {
      return cv::Mat();
    }
    cv::Mat img_RGB;
    cvtColor(input, img_RGB, cv::COLOR_RGBA2RGB);

    if (img_RGB.empty()) {
      return cv::Mat();
    }
    if (!img_RGB.data) {
      return cv::Mat();
    }
    return img_RGB;
  }
} // end namespace KSAILoaderOp
