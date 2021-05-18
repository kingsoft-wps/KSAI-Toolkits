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
#ifdef _WIN32
#include <windows.h>
#endif
#include <chrono>
#include <iostream>
#include <string>

#include "ks_ocr.h"

int main(int argc, char* argv[]) {
#ifdef _WIN32
  system("chcp 65001");
#endif
  if (argc != 6) {
    fprintf(stderr,
            "KSAI_ToolKits_OCR <detect model path> <cls model path> <recog "
            "model path> <dict path> <image path>\n");
    return 1;
  }
  const std::string detect_model_path = argv[1];
  const std::string cls_model_path = argv[2];
  const std::string recog_model_path = argv[3];
  const std::string dict_path = argv[4];
  const std::string image_path = argv[5];

  std::unique_ptr<KSAIOCR> ocr(new KSAIOCR);
  if (ocr->Init(detect_model_path, cls_model_path, recog_model_path,
                dict_path)) {
    if (image_path.empty()) {
      std::cout << "image path is invalid !!" << std::endl;
      return 1;
    }
    cv::Mat image = cv::imread(image_path);
    KSAIResults results;
    const int processNum = 1;
    for (int i = 0; i < processNum; i++) {
      auto start = std::chrono::system_clock::now();
      ocr->Process(image, results);
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cout << "finish the " << i << " process!! and cost "
                << double(duration.count()) *
                       std::chrono::microseconds::period::num /
                       std::chrono::microseconds::period::den
                << "s" << std::endl;

      // draw the detect rects and recog words on the image
      cv::Mat draw_img = image.clone();
      std::cout << "boxes size:" << results.boxes.size() << std::endl;
      for (size_t i = 0; i < results.boxes.size(); i++) {
        line(draw_img,
             cv::Point(results.boxes[i].points_[0].x_,
                       results.boxes[i].points_[0].y_),
             cv::Point(results.boxes[i].points_[1].x_,
                       results.boxes[i].points_[1].y_),
             cv::Scalar(rand() % 255, rand() % 255, rand() % 255), 5);
        line(draw_img,
             cv::Point(results.boxes[i].points_[1].x_,
                       results.boxes[i].points_[1].y_),
             cv::Point(results.boxes[i].points_[2].x_,
                       results.boxes[i].points_[2].y_),
             cv::Scalar(rand() % 255, rand() % 255, rand() % 255), 5);
        line(draw_img,
             cv::Point(results.boxes[i].points_[2].x_,
                       results.boxes[i].points_[2].y_),
             cv::Point(results.boxes[i].points_[3].x_,
                       results.boxes[i].points_[3].y_),
             cv::Scalar(rand() % 255, rand() % 255, rand() % 255), 5);
        line(draw_img,
             cv::Point(results.boxes[i].points_[3].x_,
                       results.boxes[i].points_[3].y_),
             cv::Point(results.boxes[i].points_[0].x_,
                       results.boxes[i].points_[0].y_),
             cv::Scalar(rand() % 255, rand() % 255, rand() % 255), 5);
        std::cout << results.char_results[i].c_str() << std::endl;
      }
      // show the result image
      cv::imshow("res_img", draw_img);
      cv::waitKey();
      draw_img.release();
    }
    image.release();
  }
}
