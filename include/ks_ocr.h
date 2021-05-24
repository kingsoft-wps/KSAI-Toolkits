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
#include "ks_afx.h"
#include <iostream>
#include <string>
#include <memory>

#include "opencv2/opencv.hpp"
#include "ks_data_type.h"

class KSAIOCRDetector;
class KSAIOCRAngleClassifier;
class KSAIOCRRecognizer;

class EXPORT_API KSAIOCR {
public:
  KSAIOCR();
  ~KSAIOCR();
  bool Init(const std::string &detect_model_path, const std::string &cls_model_path,
            const std::string &recog_model_path, const std::string &dict_path);
  bool Process(const cv::Mat &img, KSAIResults &results);
private:
  KSAIOCRDetector* detector_;
  KSAIOCRAngleClassifier* angleClassifier_;
  KSAIOCRRecognizer* recognizer_;
};
