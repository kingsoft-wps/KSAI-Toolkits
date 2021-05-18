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
#include "ks_ocr_cls.h"

#include <vector>

#include "base/ks_common_op.h"

bool KSAIOCRAngleClassifier::Process(const cv::Mat& in_img, cv::Mat& out_img) {
  cv::Mat mid_img;
  const int fixed_height = 32;
  KSAICommonOP::ResizeImageKeepRatio(in_img, mid_img, fixed_height);
  const int col_thresh = 400;
  if (mid_img.cols > col_thresh) {
    cv::Size dsize = cv::Size(col_thresh, fixed_height);
    cv::resize(mid_img, mid_img, dsize);
  } else {
    int pad_size = col_thresh - mid_img.cols;
    KSAICommonOP::PaddingImage(mid_img, mid_img, 0, 0, 0, pad_size);
  }
  if (mid_img.channels() == 4) {
    cv::cvtColor(mid_img, mid_img, cv::COLOR_RGBA2RGB);
  }
  const float normalize_ratio = 1.0f / 255;
  mid_img.convertTo(mid_img, CV_32FC3, normalize_ratio);
  std::vector<int> dim0 = {1, mid_img.size().height, mid_img.size().width,
                           mid_img.channels()};
  std::vector<uchar*> input_data;
  input_data.emplace_back(mid_img.data);
  std::vector<std::vector<int>> input_data_dims;
  input_data_dims.emplace_back(dim0);

  std::vector<TfLiteTensor*> output_tensor;
  std::vector<TfLiteIntArray*> output_tensor_shape;
  int ret = Inference(input_data, input_data_dims, output_tensor,
                      output_tensor_shape);
  if (ret != 1) return false;

  float* out_data = output_tensor[0]->data.f;
  int out_N = output_tensor_shape[0]->data[0];
  int out_C = output_tensor_shape[0]->data[1];
  int total_size = out_N * out_C;
  float score = 0;
  int label_idx = 0;
  for (int i = 0; i < total_size; i++) {
    if (out_data[i] > score) {
      score = out_data[i];
      label_idx = i;
    }
  }
  const int label_step = 2;
  const float score_thresh = 0.95f;
  if (label_idx % label_step == 1 && score > score_thresh) {
    cv::rotate(in_img, out_img, 1);
  } else {
    out_img = in_img;
  }

  return true;
}