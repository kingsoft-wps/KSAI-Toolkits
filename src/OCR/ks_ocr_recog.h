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
#include <opencv2/opencv.hpp>
#include <vector>

#include "base/ks_inference_base.h"
#include "ks_data_type.h"

class KSAIOCRRecognizer : public KSAIInferenceBase{
public:
  KSAIOCRRecognizer() = default;
  virtual ~KSAIOCRRecognizer() = default;
  bool Init(const std::string &model_path, const std::string &dict_path);
  bool Process(const std::vector<cv::Mat> &cls_imgs, KSAIResults &results);

private:
  bool PreProc(const cv::Mat &in_img, cv::Mat &out_img);
  bool PostProc(const std::vector<TfLiteTensor*> &tensors,
                const std::vector<TfLiteIntArray*> &tensors_shape, 
                KSAIResults &results, float &logit, std::vector<int> &label);
  int Predict(const cv::Mat &img, std::vector<TfLiteTensor*> &out_tensor,
              std::vector<TfLiteIntArray*> &out_tensor_shape);

private:
  std::map<int, std::string> char_dict_;
};
