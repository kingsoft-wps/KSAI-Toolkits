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

#include "base/ks_inference_base.h"

class KSAIOCRAngleClassifier : public KSAIInferenceBase{
public:
  KSAIOCRAngleClassifier() = default;
  virtual ~KSAIOCRAngleClassifier() = default;
  bool Process(const cv::Mat& in_img, cv::Mat& out_img);
};