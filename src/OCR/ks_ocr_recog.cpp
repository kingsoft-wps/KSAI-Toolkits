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
#include "ks_ocr_recog.h"

#include "base/ks_common_op.h"
#include "ks_ocr_utility.h"

bool KSAIOCRRecognizer::Init(const std::string &model_path,
                             const std::string &dict_path) {
  if (KSAIInferenceBase::Init(model_path) &&
      KSAIOCRUtility::CreateDict(dict_path, char_dict_)) {
    return true;
  } else {
    return false;
  }
}

bool KSAIOCRRecognizer::PreProc(const cv::Mat &in_img, cv::Mat &out_img) {
  const int fixed_height = 32;
  KSAICommonOP::ResizeImageKeepRatio(in_img, out_img, fixed_height);
  out_img.convertTo(out_img, CV_32F);
  const float max_value = 255.0;
  const float normalize_value = 0.5;
  out_img /= max_value;
  out_img -= normalize_value;
  out_img /= normalize_value;
  return true;
}

bool KSAIOCRRecognizer::PostProc(
    const std::vector<TfLiteTensor *> &tensors,
    const std::vector<TfLiteIntArray *> &tensors_shape, KSAIResults &results,
    float &logit, std::vector<int> &label) {
  int out_N = tensors_shape[0]->data[0];
  int out_W = tensors_shape[0]->data[1];
  int out_C = tensors_shape[0]->data[2];
  std::vector<int> shape{out_N, out_W, out_C};
  std::vector<std::vector<int>> output_v;
  std::vector<float> logit_v;
  KSAIOCRUtility::CTCGreedyDecoder(shape, tensors[0]->data.f, output_v,
                                   logit_v);
  logit = logit_v[0];
  label = output_v[0];

  std::string char_res = "";
  for (size_t i = 0; i < label.size(); i++) {
    char_res += char_dict_[label[i]];
  }
  results.char_results.push_back(char_res);
  return true;
}

int KSAIOCRRecognizer::Predict(
    const cv::Mat &img, std::vector<TfLiteTensor *> &out_tensor,
    std::vector<TfLiteIntArray *> &out_tensor_shape) {
  std::vector<int> dim = {1, 32, img.size().width, 1};
  std::vector<uchar *> input_data;
  input_data.emplace_back(img.data);
  std::vector<std::vector<int>> input_dims;
  input_dims.emplace_back(dim);

  int ret = Inference(input_data, input_dims, out_tensor, out_tensor_shape);
  if (ret != 1) return ret;
  return 1;
}

bool KSAIOCRRecognizer::Process(const std::vector<cv::Mat> &cls_imgs,
                                KSAIResults &results) {
  int ret = 1;
  for (int i = 0; i < cls_imgs.size(); i++) {
    cv::Mat cls_img;
    cls_imgs[i].convertTo(cls_img, CV_8UC1);
    cv::Mat out_img;
    PreProc(cls_img, out_img);
    std::vector<TfLiteTensor *> output_tensor;
    std::vector<TfLiteIntArray *> output_tensor_shape;
    ret = Predict(out_img, output_tensor, output_tensor_shape);
    if (ret != 1) return false;
    float logit = 0;
    std::vector<int> label;
    PostProc(output_tensor, output_tensor_shape, results, logit, label);
  }
  return true;
}
