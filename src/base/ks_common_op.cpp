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
#include "ks_common_op.h"

#include <iostream>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

namespace KSAICommonOP{
  void LimitMaxSide(const cv::Mat &img_in, cv::Mat &img_out,
                    const float &max_side_len) {
    cv::Size size = img_in.size();
    float width = static_cast<float>(size.width);
    float height = static_cast<float>(size.height);

    float ratio = 1.0f;
    float resize_h = height;
    float resize_w = width;
    int max_len = static_cast<int>(std::max(resize_h, resize_w));
    if (max_len > max_side_len) {
      ratio = max_len / max_side_len;
      resize_w = resize_w / ratio;
      resize_h = resize_h / ratio;
    }
    int iresize_h = static_cast<int>(resize_h);
    int iresize_w = static_cast<int>(resize_w);
    cv::Size dsize = cv::Size(iresize_w, iresize_h);
    cv::resize(img_in, img_out, dsize);
  }

  // Keep the aspect ratio to resize the image
  void ResizeImageKeepRatio(const cv::Mat &img_in,
                            cv::Mat &img_out, const int &fixed_height) {
    int width = img_in.size().width;
    int height = img_in.size().height;
    float ratio = static_cast<float>(height * 1.0 / fixed_height);
    int resize_width = static_cast<int>(width / ratio);
    cv::Size dsize = cv::Size(resize_width, fixed_height);
    cv::resize(img_in, img_out, dsize);
  }

  // Resize image so that its length and width are multiples of 32
  void ResizeImageWithMultiple32(const cv::Mat &img_in, cv::Mat &img_out) {
    cv::Size size = img_in.size();
    float resize_w = static_cast<float>(size.width);
    float resize_h = static_cast<float>(size.height);
    const int multiple = 32;

    int iresize_h = static_cast<int>(resize_h);
    int iresize_w = static_cast<int>(resize_w);

    iresize_h = iresize_h % multiple == 0 ? iresize_h : ((iresize_h / multiple) + 1) * multiple;
    iresize_w = iresize_w % multiple == 0 ? iresize_w : ((iresize_w / multiple) + 1) * multiple;

    cv::Size dsize = cv::Size(iresize_w, iresize_h);
    resize(img_in, img_out, dsize);
  }

  bool NeedInverseColor(const cv::Mat &img) {
    cv::Mat dst;
    img.convertTo(dst, CV_8UC1);
    const double thresh = 0;
    const double max_value = 255;
    cv::threshold(dst, dst, thresh, max_value, cv::THRESH_OTSU);
    const int top = 10;
    const int bottom = 10;
    const int left = 10;
    const int right = 10;
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_REPLICATE);
    const int gray_value = 128;
    dst.forEach<unsigned char>(
        [&](unsigned char &pixel, const int position[]) -> void {
          pixel = (pixel <= gray_value ? 0 : 1);
        });

    cv::Scalar ratio = sum(dst) * 1 / (dst.size().width * dst.size().height);
    const float ratio_thresh = 0.5;
    return ratio[0] < ratio_thresh;
  }

  void PaddingImage(const cv::Mat &img_in, cv::Mat &img_out,
                    const int &top, const int &bottom,
                    const int &left, const int &right) {
    cv::copyMakeBorder(img_in, img_out, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
  }
} // end namespace KSAICommonOP
