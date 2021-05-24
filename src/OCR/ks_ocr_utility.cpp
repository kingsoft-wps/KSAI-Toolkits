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
#include "ks_ocr_utility.h"

#include <algorithm>
#include <fstream>
#include <vector>

#include "clipper/clipper.hpp"
#include "ks_afx.h"
#include "ks_line.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

namespace KSAIOCRUtility {
// Read character dictionary from txt file
bool CreateDict(const std::string &dict_path,
                std::map<int, std::string> &char_dict) {
  std::ifstream in(dict_path);
  if (!in.is_open()) {
    std::cout << "dict_path is invalid!!" << std::endl;
    return false;
  }
  int idx = 0;
  while (!in.eof()) {
    std::string line;
    std::getline(in, line);
    std::string::size_type pos = line.find('\r');
    std::string sub_str = line.substr(0, pos);
    char_dict.emplace(std::make_pair(idx, sub_str));
    ++idx;
  }
  in.close();
  return true;
}

void MergeHorizontal(const VOCRectf &rects, const std::vector<std::vector<int>> &labels,
                     const std::vector<std::string> &results,
                     std::vector<std::vector<int>> &duanluo_results_labels,
                     std::vector<std::string> &merged_results) {
  std::list<Line> result;

  for (size_t i = 0; i < results.size(); i++) {
    Line temp = Line(rects[i], labels[i], results[i]);
    result.push_back(temp);
  }

  std::list<Line> lines;
  std::list<std::list<Line>> paras;
  while (result.size()) {
    bool end_flag = false;
    Line current_line = result.front();
    result.pop_front();
    lines.push_back(current_line);
    while (!end_flag) {
      if (result.size() == 0) {
        end_flag = true;
        paras.push_back(lines);
        lines.clear();
        break;
      }
      Line nearest_line = current_line.FindNearest(result);
      if (current_line.CoParagraph(nearest_line)) {
        lines.push_back(nearest_line);
        result.remove(nearest_line);
        current_line = nearest_line;
      } else {
        end_flag = true;
        paras.push_back(lines);
        lines.clear();
      }
    }
  }

  for (auto it = paras.begin(); it != paras.end(); ++it) {
    std::string seg = "";
    std::vector<int> temp_duanluo_label;
    for (auto line = it->begin(); line != it->end(); line++) {
      seg += line->chars_;
      for (size_t j = 0; j < line->label_.size(); j++) {
        temp_duanluo_label.push_back(line->label_[j]);
      }
    }
    duanluo_results_labels.push_back(temp_duanluo_label);
    merged_results.push_back(seg);
  }
}
bool CTCGreedyDecoder(const std::vector<int> &shape, const float* input,
                      std::vector<std::vector<int>> &output,
                      std::vector<float> &logit) {
  int batch_size = shape[0];
  int width = shape[1];
  int num_classes = shape[2];
  int blank_index = num_classes - 1;
  float score_sum = 0;
  for (int b = 0; b < batch_size; b++) {
    std::vector<int> max_class_idx_w;
    int prev_class_ix = -1;
    for (int w = 0; w < width; w++) {
      int max_class_idx = 0;
      float max_class_val = -10000;
      int inputIdx = (b * width + w) * num_classes;
      for (int c = 0; c < num_classes; c++) {
        if (max_class_val < input[inputIdx + c]) {
          max_class_idx = c;
          max_class_val = input[inputIdx + c];
        }
      }
      if ((max_class_idx != blank_index) && !(max_class_idx == prev_class_ix)) {
        max_class_idx_w.emplace_back(max_class_idx);
      }
      prev_class_ix = max_class_idx;
      score_sum += max_class_val;
    }

    logit.emplace_back(score_sum);
    score_sum = 0;
    output.emplace_back(max_class_idx_w);
  }
  return 1;
}

std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(
    const cv::Mat &pred, const cv::Mat &bitmap, const float &box_thresh,
    const float &unclip_ratio, const int &origin_w, const int &origin_h) {
  const int max_candidates = 500;
  float min_size = 3.0;

  int width = bitmap.cols;
  int height = bitmap.rows;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);
  int num_contours =
      contours.size() >= max_candidates ? max_candidates : static_cast<int>(contours.size());

  std::vector<std::vector<std::vector<int>>> boxes;

  for (int _i = 0; _i < num_contours; _i++) {
    if (contours[_i].size() <= 2) {
      continue;
    }
    std::vector<cv::Point> contours_poly(
        contours[_i].size());  //用于存放折线点集
    float epsilon = static_cast<float>(0.001 * arcLength(contours[_i], true));
    cv::approxPolyDP(cv::Mat(contours[_i]), contours_poly, epsilon, true);
    float score;
    std::vector<std::vector<float>> box_array;
    for (int k = 0; k < contours_poly.size(); k++) {
      std::vector<float> tmp_box;
      tmp_box.push_back(static_cast<float>(contours_poly[k].x));
      tmp_box.push_back(static_cast<float>(contours_poly[k].y));
      box_array.push_back(tmp_box);
    }

    cv::RotatedRect box = cv::minAreaRect(contours_poly);
    // cv::RotatedRect box = cv::minAreaRect(contours[_i]);
    auto filter_box = GetMiniBoxes(box, min_size);
    if (filter_box.size() == 0) continue;
 
    // auto box_for_unclip = array;
    // end get_mini_box
    score = BoxScoreFast(box_array, pred);

    if (score < box_thresh) continue;

    // start for unclip
    cv::RotatedRect unclip_box = UnClip(filter_box, unclip_ratio);
    if (unclip_box.size.height <= 1.0 && unclip_box.size.width <= 1.0) {
      continue;
    }
    // end for unclip

    // cv::RotatedRect clipbox = points;
    auto filter_unclip = GetMiniBoxes(unclip_box, min_size);
    if (filter_unclip.size() == 0) continue;

    // int dest_width = pred.cols;
    // int dest_height = pred.rows;
    std::vector<std::vector<int>> clip_box_array;

    for (int num_pt = 0; num_pt < 4; num_pt++) {
      std::vector<int> a{
          int(Clampf(
              roundf(filter_unclip[num_pt][0] / float(width) * float(width)), 0,
              float(origin_w))),
          int(Clampf(
              roundf(filter_unclip[num_pt][1] / float(height) * float(height)),
              0, float(origin_h)))};
      clip_box_array.push_back(a);
    }
    boxes.push_back(clip_box_array);

  }  // end for
  return boxes;
}

std::vector<std::vector<float>> GetMiniBoxes(const cv::RotatedRect &box, const float &min_size) {
  float ssid = std::max(box.size.width, box.size.height);
  if (ssid < min_size) {
    std::vector<std::vector<float>> mini_box;
    return mini_box;
  }
  cv::Mat points;
  cv::boxPoints(box, points);

  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), XsortFp32);

  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;

  return array;
}

cv::RotatedRect UnClip(const std::vector<std::vector<float>> &box,
                       const float &unclip_ratio) {
  float distance = 1.0;

  GetContourArea(box, unclip_ratio, distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint(int(box[0][0]), int(box[0][1]))
    << ClipperLib::IntPoint(int(box[1][0]), int(box[1][1]))
    << ClipperLib::IntPoint(int(box[2][0]), int(box[2][1]))
    << ClipperLib::IntPoint(int(box[3][0]), int(box[3][1]));
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);
  std::vector<cv::Point2f> points;

  for (int j = 0; j < soln.size(); j++) {
    for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
      points.emplace_back(soln[j][i].X, soln[j][i].Y);
    }
  }
  cv::RotatedRect res;
  if (points.size() <= 0) {
    res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
  } else {
    res = cv::minAreaRect(points);
  }
  return res;
}

void GetContourArea(const std::vector<std::vector<float>> &box,
                    const float &unclip_ratio, float &distance) {
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = fabs(float(area / 2.0));

  distance = area * unclip_ratio / dist;
}

std::vector<std::vector<float>> Mat2Vector(const cv::Mat &mat) {
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

bool XsortFp32(const std::vector<float> &a, const std::vector<float> &b) {
  if (a[0] != b[0]) return a[0] < b[0];
  return false;
}

bool XsortInt(const std::vector<int> &a, const std::vector<int> &b) {
  if (a[0] != b[0]) return a[0] < b[0];
  return false;
}

float BoxScoreFast(const std::vector<std::vector<float>> &box_array,
                   const cv::Mat &pred) {
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;
  int num_pt = static_cast<int>(box_array.size());
  std::vector<float> box_x;
  std::vector<float> box_y;
  for (int i = 0; i < num_pt; i++) {
    box_x.push_back(array[i][0]);
    box_y.push_back(array[i][1]);
  }
  int xmin = Clamp<int>(
      int(std::floor(*(std::min_element(box_x.begin(), box_x.end())))), 0,
      width - 1);
  int xmax = Clamp<int>(
      int(std::ceil(*(std::max_element(box_x.begin(), box_x.end())))), 0,
      width - 1);
  int ymin = Clamp<int>(
      int(std::floor(*(std::min_element(box_y.begin(), box_y.end())))), 0,
      height - 1);
  int ymax = Clamp<int>(
      int(std::ceil(*(std::max_element(box_y.begin(), box_y.end())))), 0,
      height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
  cv::Point *root_point = new cv::Point[num_pt];
  for (int p = 0; p < num_pt; p++) {
    root_point[p] = cv::Point(int(array[p][0]) - xmin, int(array[p][1]) - ymin);
  }
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {num_pt};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));
  RELEASE_ARR(root_point);
  cv::Mat cropped_img;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(cropped_img);

  auto score = cv::mean(cropped_img, mask)[0];
  return static_cast<float>(score);
}

void GetWarpImgs(const cv::Mat &img, const VOCRectf &rects,
                 std::vector<cv::Mat> &warp_imgs) {
  for (size_t i = 0; i < rects.size(); ++i) {
    OCRectf rect = rects[i];
    cv::Point2f p[] = {cv::Point2f(rect.points_[0].x_, rect.points_[0].y_),
                       cv::Point2f(rect.points_[1].x_, rect.points_[1].y_),
                       cv::Point2f(rect.points_[2].x_, rect.points_[2].y_),
                       cv::Point2f(rect.points_[3].x_, rect.points_[3].y_)};

    float w = rect.points_[1].x_ - rect.points_[0].x_;
    float h = rect.points_[2].y_ - rect.points_[1].y_;

    cv::Point2f p1(0, 0), p2(w, 0), p3(w, h), p4(0, h);
    cv::Point2f dst[4] = {p1, p2, p3, p4};
    img.convertTo(img, CV_8UC1);
    cv::Mat trans = getPerspectiveTransform(p, dst);
    cv::Mat warped, gray;

    warpPerspective(img, warped, trans,
                    cv::Size(static_cast<int>(w), static_cast<int>(h)));
    warp_imgs.push_back(warped);
  }
}
}  // end namespace KSAIOCRUtility
