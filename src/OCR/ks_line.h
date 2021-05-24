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
#include <cmath>
#include <list>
#include <string>
#include <vector>

#include "ks_data_type.h"
class Line {
public:
  Line(const OCRectf &rect, const std::vector<int> &label, const std::string &result) {
    label_ = label;
    chars_ = result;
    coord_ = rect;
    end_flag_ = false;

    w_ = static_cast<float>(std::sqrt(std::pow((rect.points_[0].x_ - rect.points_[1].x_), 2) +
                            std::pow((rect.points_[0].y_ - rect.points_[1].y_), 2)));
    h_ = static_cast<float>(std::sqrt(std::pow((rect.points_[3].x_ - rect.points_[1].x_), 2) +
                            std::pow((rect.points_[3].y_ - rect.points_[1].y_), 2)));
  }

  Line &operator=(const Line &other) {
    label_ = other.label_;
    chars_ = other.chars_; // note: shallow copy
    coord_ = other.coord_;
    end_flag_ = other.end_flag_;
    w_ = other.w_;
    h_ = other.h_;
    return *this;
  }

  Line(const Line &other) {
    label_ = other.label_;
    chars_ = other.chars_; // note: shallow copy
    coord_ = other.coord_;
    end_flag_ = other.end_flag_;
    w_ = other.w_;
    h_ = other.h_;
  }
  bool operator==(const Line &other) const {
    return coord_ == other.coord_ && !chars_.compare(other.chars_);
  }

  float Distance(const OCRectf &other) {
    return static_cast<float>(std::sqrt(std::pow((coord_.points_[2].x_ - other.points_[0].x_), 2) +
                              std::pow((coord_.points_[2].y_ - other.points_[0].y_), 2)));
  }

  bool CoParagraph(Line &other) {
    static const float h_ratio = 0.1f;
    static const float w_ratio = 0.15f;
    static const float w_factor = 2.f;
    static const float h_factor = 1.0f;
    static const float dis_ratio = 2.5f;

    if (end_flag_) {
      return false;
    }
    float hr = h_ / other.h_;
    float wr = w_ / other.w_;
    float wd = w_ - other.w_;
    float end = coord_.points_[3].x_ - other.coord_.points_[1].x_;
    float hgap = other.coord_.points_[1].y_ - coord_.points_[3].y_;
    if (label_.size() == 0) {
      return false;
    }

    if (hgap > h_) {
      return false;
    }

    if (hr > 1 + h_ratio || hr < 1 - h_ratio) {
      return false;
    }
    if (end < -w_factor * h_) {
      return false;
    }
    if (wd > w_factor * h_) {
      other.end_flag_ = true;
    }
    float dis = Distance(other.coord_);

    if (dis > dis_ratio * h_) {
      return false;
    }
    return true;
  }

  Line FindNearest(const std::list<Line> &others) {
    Line ret = others.front();
    float nearest = 1e9;
    for (auto other = others.begin(); other != others.end(); other++) {
      float dis = Distance((*other).coord_);
      if (nearest > dis) {
        nearest = dis;
        ret = *other;
      }
    }
    return ret;
  }

  std::string chars_;
  std::vector<int> label_;
  OCRectf coord_;
  bool end_flag_;
  float w_;
  float h_;
};
