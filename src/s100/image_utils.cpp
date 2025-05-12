// Copyright (c) 2025，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "include/image_utils.h"

std::shared_ptr<DNNTensor> ImageUtils::GetBGRTensorFromBGR(
                                          const cv::Mat &bgr_mat_tmp,                                                     
                                          int scaled_img_height,
                                          int scaled_img_width,
                                          hbDNNTensorProperties &tensor_properties) {
  cv::Mat bgr_mat;
  bgr_mat_tmp.copyTo(bgr_mat);
  cv::Mat pixel_values_mat;
  auto w_stride = ALIGN_16(scaled_img_width);
  int channel = 3;
  int src_elem_size = 4;
  int original_img_width = bgr_mat.cols;
  int original_img_height = bgr_mat.rows;

  cv::Mat pad_frame;
  if (static_cast<uint32_t>(original_img_width) != w_stride ||
    original_img_height != scaled_img_height) {
    pad_frame = cv::Mat(scaled_img_height, w_stride, CV_8UC3, cv::Scalar::all(0));
    if (static_cast<uint32_t>(original_img_width) > w_stride ||
      original_img_height > scaled_img_height) {
      float ratio_w = static_cast<float>(original_img_width) / static_cast<float>(w_stride);
      float ratio_h = static_cast<float>(original_img_height) / static_cast<float>(scaled_img_height);
      float dst_ratio = std::max(ratio_w, ratio_h);
      uint32_t resized_width = static_cast<float>(original_img_width) / dst_ratio;
      uint32_t resized_height = static_cast<float>(original_img_height) / dst_ratio;
      cv::resize(bgr_mat, bgr_mat, cv::Size(resized_width, resized_height));
    }
    // 复制到目标图像到起点
    bgr_mat.copyTo(pad_frame(cv::Rect(0, 0, bgr_mat.cols, bgr_mat.rows)));
  } else {
    pad_frame = bgr_mat;
  }

  cv::Mat mat_tmp;
  pad_frame.convertTo(mat_tmp, CV_32F); 
  cv::cvtColor(mat_tmp, mat_tmp, cv::COLOR_BGR2RGB);
  mat_tmp /= 255.0;

  cv::Scalar mean(0.485, 0.456, 0.406);  // BGR 通道均值
  cv::Scalar std(0.229, 0.224, 0.225);   // BGR 通道标准差

  // 按通道减去均值，再除以标准差
  std::vector<cv::Mat> channels(3);
  cv::split(mat_tmp, channels);  // 分离通道

  for (int i = 0; i < 3; i++) {
    channels[i] = (channels[i] - mean[i]) / std[i];
  }

  // 合并通道
  cv::merge(channels, pixel_values_mat);

  auto *mem = new hbUCPSysMem;
  hbUCPMallocCached(mem, scaled_img_height * w_stride * channel * src_elem_size, 0);

  uint8_t *data = pixel_values_mat.data;
  auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem->virAddr);

  for (int h = 0; h < scaled_img_height; ++h) {
    for (int w = 0; w < scaled_img_width; ++w) {
      for (int c = 0; c < channel; ++c) {
        auto *raw = hb_mem_addr + c * scaled_img_height * w_stride * src_elem_size + h * w_stride * src_elem_size + w * src_elem_size;
        auto *src = data + h * scaled_img_width * channel * src_elem_size + w * channel * src_elem_size + c * src_elem_size;
        memcpy(raw, src, src_elem_size);
      }
    }
  }

  hbUCPMemFlush(mem, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new DNNTensor;
  input_tensor->properties = tensor_properties;
  input_tensor->sysMem.virAddr = reinterpret_cast<void *>(mem->virAddr);
  input_tensor->sysMem.phyAddr = mem->phyAddr;
  input_tensor->sysMem.memSize = scaled_img_height * scaled_img_width * channel * src_elem_size;

  return std::shared_ptr<DNNTensor>(
      input_tensor, [mem](DNNTensor *input_tensor) {
        // Release memory after deletion
        hbUCPFree(mem);
        delete mem;
        delete input_tensor;
      });
}
