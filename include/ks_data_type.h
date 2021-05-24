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
#include <assert.h>
#include <vector>

template <typename T> class MPoint {
public:
  MPoint() : x_(0), y_(0) {}
  MPoint(T x, T y) : x_(x), y_(y) {}
  MPoint(const MPoint<T> &p) {
    x_ = p.x_;
    y_ = p.y_;
  }
  MPoint(MPoint<T> &&p) {
    x_ = p.x_;
    y_ = p.y_;
  }
  MPoint<T> &operator=(const MPoint<T> &p) {
    x_ = p.x_;
    y_ = p.y_;
    return *this;
  }

  bool operator==(const MPoint<T> &p) const { return x_ == p.x_ && y_ == p.y_; }

  T x_;
  T y_;
};

template <typename T> class OCRect {
public:
  OCRect() {}
  OCRect(const std::vector<MPoint<T>> &p) {
    assert(p.size() == 4);
    points_[0] = p[0];
    points_[1] = p[1];
    points_[2] = p[2];
    points_[3] = p[3];
  }
  OCRect(const std::vector<MPoint<T>> &&p) {
    assert(p.size() == 4);
    points_[0] = p[0];
    points_[1] = p[1];
    points_[2] = p[2];
    points_[3] = p[3];
  }
  OCRect(const OCRect<T> &r) {
    points_[0] = r.points_[0];
    points_[1] = r.points_[1];
    points_[2] = r.points_[2];
    points_[3] = r.points_[3];
  }
  OCRect(const OCRect<T> &&r) {
    points_[0] = r.points_[0];
    points_[1] = r.points_[1];
    points_[2] = r.points_[2];
    points_[3] = r.points_[3];
  }

  OCRect<T> &operator=(const OCRect<T> &r) {
    points_[0] = r.points_[0];
    points_[1] = r.points_[1];
    points_[2] = r.points_[2];
    points_[3] = r.points_[3];
    return *this;
  }

  bool operator==(const OCRect<T> &r) const {
    return points_[0] == r.points_[0] && points_[1] == r.points_[1] &&
           points_[2] == r.points_[2] && points_[3] == r.points_[3];
  }

  /**
          0---------------1
          |               |
          |               |
          3---------------2
  **/
  MPoint<T> points_[4];
};

typedef MPoint<float> MPointf;
typedef OCRect<float> OCRectf;
typedef std::vector<OCRectf> VOCRectf;

typedef struct KSAIResults {
  std::vector<std::string> char_results;
  VOCRectf boxes;
} KSAIResults;


