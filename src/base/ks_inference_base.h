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
#include <iostream>
#include <vector>
#include <memory>

#include "ksai-lite/c_api.h"

class KSAIInferenceBase {
public:
  KSAIInferenceBase();
  virtual ~KSAIInferenceBase();

  virtual bool Init(const std::string &model_path);
  int Inference(const std::vector<unsigned char*> &input_data,
                const std::vector<std::vector<int>> &input_dims,
                std::vector<TfLiteTensor*> &output_tensor,
                std::vector<TfLiteIntArray*> &output_tensor_shape);  
private:
  int AllocateInputTensor(const std::vector<unsigned char*> &input_data,
                          const std::vector<std::vector<int>> &input_dims);
  void GetOutputTensor(const int &output_index, TfLiteTensor* &out_tensor,
                       TfLiteIntArray* &out_tensor_shape);
private:
  TfLiteInterpreter* interpreter_;
};
