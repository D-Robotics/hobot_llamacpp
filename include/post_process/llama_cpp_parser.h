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

#ifndef LLAMA_OUTPUT_PARSER_H_
#define LLAMA_OUTPUT_PARSER_H_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "dnn_node/dnn_node_data.h"
#include "dnn_node/util/output_parser/perception_common.h"
#include "dnn_node/util/output_parser/detection/nms.h"
#include "std_msgs/msg/string.hpp"

#include "include/cli.h"

using hobot::dnn_node::output_parser::Bbox;
using hobot::dnn_node::output_parser::Detection;
using hobot::dnn_node::output_parser::DnnParserResult;
using hobot::dnn_node::output_parser::Perception;
using hobot::dnn_node::DNNTensor;

class LlamaCppParser {
 public:
  LlamaCppParser(const std::string& model_name, const std::string& system_prompt, const int n_threads);
  ~LlamaCppParser();

  int32_t Init(const std::string &system_prompt);

  int32_t Parse(
      const std::string &user_prompt,
      std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
      std::string &result,
      rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher);

  struct llava_image_embed * GetEmbedding(std::vector<std::shared_ptr<DNNTensor>> &tensors);

 private:
  common_params params;
  struct llama_model * model_ = nullptr;
  struct llava_context * ctx_llava_ = nullptr;

};

#endif  // LLAMA_OUTPUT_PARSER_H_
