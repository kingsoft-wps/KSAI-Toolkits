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
#include "ks_inference_base.h"

#include "ks_afx.h"

#include <algorithm>

KSAIInferenceBase::KSAIInferenceBase() : interpreter_(nullptr) {}

KSAIInferenceBase::~KSAIInferenceBase() {
  if (!interpreter_) TfLiteInterpreterDelete(interpreter_);
}

bool KSAIInferenceBase::Init(const std::string &model_path)  {
  if (model_path.empty()) {
    std::cout << "model_path is invalid !!" << std::endl;
    return false;
  }
  TfLiteModel* model = TfLiteModelCreateFromFile(model_path.data());
  if (!model) {
    std::cout << "fail to create model !!" << std::endl;
    return false;
  }
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  if (!options) {
    std::cout << "fail to create options !!" << std::endl;
    return false;
  }
  interpreter_ = TfLiteInterpreterCreate(model, options);
  if (!interpreter_) {
    std::cout << "fail to create interpreter !!" << std::endl;
    return false;
  }
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  return true;
}

int KSAIInferenceBase::Inference(const std::vector<unsigned char*> &input_data,
                                 const std::vector<std::vector<int>> &input_dims,
                                 std::vector<TfLiteTensor*> &output_tensor,
                                 std::vector<TfLiteIntArray*> &output_tensor_shape){
  int ret = 1;
  ret = AllocateInputTensor(input_data, input_dims);
  if (ret != 1) return ret;
  if (TfLiteInterpreterInvoke(interpreter_) != kTfLiteOk) {
    std::cout << "fail to run invoke !!" << std::endl;
    return -4;
  }
  int output_count = TfLiteInterpreterGetOutputTensorCount(interpreter_); 
  for (int i = 0; i < output_count; i++) {
    TfLiteTensor* out_tensor;
    TfLiteIntArray* out_tensor_shape;
    GetOutputTensor(i, out_tensor, out_tensor_shape);
    output_tensor.push_back(out_tensor);
    output_tensor_shape.push_back(out_tensor_shape);
  }
  return 1;
}

int KSAIInferenceBase::AllocateInputTensor(const std::vector<unsigned char*> &input_data,
                                           const std::vector<std::vector<int>> &input_dims) {
  if (input_data.empty()) {
    std::cout << "input data is empty !!" << std::endl;
    return -1;
  }
  if (!interpreter_) {
    std::cout << "interpreter is null !!" << std::endl;
    return -2;
  }

  for (size_t i = 0; i < input_data.size(); i++) {
    int input_data_size = static_cast<int>(input_dims[i].size());
    int* input_dim = new int[input_data_size];

    int data_size = 1;
    for (size_t j = 0; j < input_data_size; j++) {
      input_dim[j] = input_dims[i][j];
      data_size *= input_dims[i][j];
    }
    TfLiteInterpreterResizeInputTensor(interpreter_, i, input_dim,
                                       static_cast<int32_t>(input_dims[i].size()));
    RELEASE_ARR(input_dim);
  }

  if (TfLiteInterpreterAllocateTensors(interpreter_) != kTfLiteOk) {
    std::cout << "fail to allocate tensors !!" << std::endl;
    return -3;
  }

  for (int i = 0; i < input_data.size(); i++) {
    TfLiteTensor* input_tensor =
        TfLiteInterpreterGetInputTensor(interpreter_, i);
    int input_data_size = static_cast<int>(input_dims[i].size());
    int data_size = 1;
    for (size_t j = 0; j < input_data_size; j++) {
      data_size *= input_dims[i][j];
    }

    int copy_bytes = data_size * sizeof(float);
    std::copy_n(input_data[i], copy_bytes, input_tensor->data.raw);
  }
  return 1;
}

void KSAIInferenceBase::GetOutputTensor(const int &output_index, TfLiteTensor *&out_tensor,
                                        TfLiteIntArray *&out_tensor_shape) {
  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter_, output_index);
  out_tensor = const_cast<TfLiteTensor *>(output_tensor);
  out_tensor_shape = output_tensor->dims;
}