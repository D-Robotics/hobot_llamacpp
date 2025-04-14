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

#ifndef CLI_H_
#define CLI_H_

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "base64.hpp"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "src/llama-context.h"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

struct llava_context {
 struct clip_ctx * ctx_clip = NULL;
 struct llama_context * ctx_llama = NULL;
 struct llama_model * model = NULL;
};

struct llava_image_embed {
  float * embed;
  int n_image_pos;
};

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

class CLI {
 public:
  static bool internvl2_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                         int n_batch, int * n_past, int * st_pos_id);
  static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past, int * st_pos_id);
  static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past, int * st_pos_id);
  static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, int * st_pos_id, bool add_bos);
  static const char * sample(struct common_sampler * smpl,
                             struct llama_context * ctx_llama,
                             int * n_past, int * st_pos_id);
  static void process_system_prompt(struct llava_context * ctx_llava, common_params * params, const std::string & sprompt);
  static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, common_params * params, const std::string & prompt, std::string &response, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher);
  static struct llama_model * llava_init(common_params * params);
  static struct llava_context * llava_init_context(common_params * params, llama_model * model);
  static void llava_free(struct llava_context * ctx_llava);
};

#endif  // CLI_H_