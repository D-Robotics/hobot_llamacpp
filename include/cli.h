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
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "base64.hpp"
#include "ggml.h"
#include "llama.h"
#include "common/log.h"
#include "sampling.h"
#include "src/llama-context.h"
#include "common.h"
#include "console.h"
#include "chat.h"
#include "arg.h"

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

static const char * DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant";

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
  (void) argc;

  LOG("\nexample usage:\n");
  LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
  LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
  LOG("\n");
}

bool isChinese(const std::string& str, size_t i);
bool isChinesePunctuation(const std::string& str, size_t i);
std::string filterChineseAndPunctuation(const std::string& input, bool& hasChinese, bool& hasPunctuation);


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