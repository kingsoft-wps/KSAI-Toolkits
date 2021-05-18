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
#include "ks_ocr_det.h"

#include <fstream>
#include <iostream>

#include "base/ks_common_op.h"
#include "ks_ocr_utility.h"
#define SAT(a, b, c) (b > a ? (b < c ? b : c) : a)

bool KSAIOCRDetector::Process(const cv::Mat& img, VOCRectf& rects) {
  int origin_w = img.size().width;
  int origin_h = img.size().height;
  cv::Mat img_out;
  const float max_len = 800;
  KSAICommonOP::LimitMaxSide(img, img_out, max_len);
  KSAICommonOP::ResizeImageWithMultiple32(img_out, img_out);
  if (img_out.channels() == 4) {
    cv::cvtColor(img_out, img_out, cv::COLOR_RGBA2RGB);
  }

  float ratio_w = static_cast<float>(img_out.size().width) / origin_w;
  float ratio_h = static_cast<float>(img_out.size().height) / origin_h;

  cv::Mat convert_img;
  img_out.convertTo(convert_img, CV_32FC3);
  std::vector<int> dim0 = {1, convert_img.size().height,
                           convert_img.size().width, convert_img.channels()};
  std::vector<uchar*> inputData;
  inputData.emplace_back(convert_img.data);
  std::vector<std::vector<int>> inputDataDims;
  inputDataDims.emplace_back(dim0);

  std::vector<TfLiteTensor*> output_tensor;
  std::vector<TfLiteIntArray*> output_tensor_shape;
  int ret =
      Inference(inputData, inputDataDims, output_tensor, output_tensor_shape);
  if (ret != 1) return false;

  const double max_value = 255;
  const double threshold = 0.3 * 255.0;
  float box_thresh = 0.4f;
  float unclip_ratio = 1.4f;
  float* out_data = output_tensor[0]->data.f;
  int out_H = output_tensor_shape[0]->data[1];
  int out_W = output_tensor_shape[0]->data[2];
  int total_size = out_H * out_W;
  std::vector<float> pred(total_size, 0.0);
  std::vector<unsigned char> cbuf(total_size, ' ');
  for (int i = 0; i < total_size; i++) {
    pred[i] = float(out_data[i]);
    cbuf[i] = (unsigned char)((out_data[i]) * max_value);
  }
  cv::Mat cbuf_map(out_H, out_W, CV_8UC1, (unsigned char*)cbuf.data());
  cv::Mat prob_map(out_H, out_W, CV_32F, (float*)pred.data());

  cv::Mat bit_map;
  cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);

  cv::Mat dilated_map;
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(bit_map, dilated_map, element);

  std::vector<std::vector<std::vector<int>>> boxes;
  boxes = KSAIOCRUtility::BoxesFromBitmap(prob_map, dilated_map, box_thresh,
                                          unclip_ratio, origin_w, origin_h);
  for (int i = 0; i < boxes.size(); i++) {
    for (int j = 0; j < boxes[i].size(); j++) {
      boxes[i][j][0] = static_cast<int>(boxes[i][j][0] / ratio_w);
      boxes[i][j][1] = static_cast<int>(boxes[i][j][1] / ratio_h);
    }
  }

  for (int i = 0; i < boxes.size(); ++i) {
    std::vector<cv::Range> tempmat;
    tempmat.emplace_back(cv::Range(i, i + 1));
    tempmat.emplace_back(cv::Range::all());

    std::vector<MPoint<float>> tempvec;
    tempvec.emplace_back(MPoint<float>(SAT(0.0f, boxes[i][0][0], origin_w),
                                       SAT(0.0f, boxes[i][0][1], origin_h)));
    tempvec.emplace_back(MPoint<float>(SAT(0.0f, boxes[i][1][0], origin_w),
                                       SAT(0.0f, boxes[i][1][1], origin_h)));
    tempvec.emplace_back(MPoint<float>(SAT(0.0f, boxes[i][2][0], origin_w),
                                       SAT(0.0f, boxes[i][2][1], origin_h)));
    tempvec.emplace_back(MPoint<float>(SAT(0.0f, boxes[i][3][0], origin_w),
                                       SAT(0.0f, boxes[i][3][1], origin_h)));
    rects.emplace_back(OCRect<float>(tempvec));

    std::sort(rects.begin(), rects.end(),
              [](const OCRectf& rect1, const OCRectf& rect2) -> bool {
                return rect1.points_[0].y_ < rect2.points_[0].y_;
              });
  }
  return true;
}
