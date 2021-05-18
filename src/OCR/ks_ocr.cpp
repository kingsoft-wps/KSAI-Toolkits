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
#include "ks_ocr.h"

#include <fstream>
#include <iostream>

#include "OCR/ks_ocr_cls.h"
#include "OCR/ks_ocr_det.h"
#include "OCR/ks_ocr_recog.h"
#include "ks_ocr_utility.h"

KSAIOCR::KSAIOCR()
    : detector_(nullptr), angleClassifier_(nullptr), recognizer_(nullptr) {}

KSAIOCR::~KSAIOCR() {
  if (detector_) {
    delete detector_;
    detector_ = nullptr;
  }
  if (angleClassifier_) {
    delete angleClassifier_;
    angleClassifier_ = nullptr;
  }
  if (recognizer_) {
    delete recognizer_;
    recognizer_ = nullptr;
  }
}

bool KSAIOCR::Init(const std::string detect_model_path,
                   const std::string cls_model_path,
                   const std::string recog_model_path,
                   const std::string dict_path) {
  if (detector_ == nullptr) {
    detector_ = new KSAIOCRDetector();
  }
  if (angleClassifier_ == nullptr) {
    angleClassifier_ = new KSAIOCRAngleClassifier();
  }
  if (recognizer_ == nullptr) {
    recognizer_ = new KSAIOCRRecognizer();
  }

  if (detector_->Init(detect_model_path) &&
      angleClassifier_->Init(cls_model_path) &&
      recognizer_->Init(recog_model_path, dict_path)) {
    return true;
  } else {
    return false;
  }
}

bool KSAIOCR::Process(const cv::Mat& img, KSAIResults& results) {
  int ret = true;
  VOCRectf rects;
  ret = detector_->Process(img, rects);
  if (ret == false) return false;
  results.boxes = rects;
  std::vector<cv::Mat> warp_imgs;
  KSAIOCRUtility::GetWarpImgs(img, rects, warp_imgs);
  std::vector<cv::Mat> cls_imgs;
  cv::Mat cls_img;
  const float scale = 1.3f;
  for (size_t i = 0; i < warp_imgs.size(); i++) {
    int w = static_cast<int>(warp_imgs[i].size().width);
    int h = static_cast<int>(warp_imgs[i].size().height);
    if (h > static_cast<int>(scale * w)) {
      transpose(warp_imgs[i], warp_imgs[i]);
      flip(warp_imgs[i], warp_imgs[i], 0);
    }
    ret = angleClassifier_->Process(warp_imgs[i], cls_img);
    if (ret == false) return false;
    cvtColor(cls_img, cls_img, cv::COLOR_RGB2GRAY);
    cls_imgs.push_back(cls_img);
  }
  ret = recognizer_->Process(cls_imgs, results);
  if (ret == false) return false;

  return true;
}
