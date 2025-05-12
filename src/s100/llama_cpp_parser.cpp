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

#include "include/post_process/llama_cpp_parser.h"

LlamaCppParser::LlamaCppParser(const std::string& model_name, const std::string& system_prompt, const int n_threads) {
  common_init();
  params.model = model_name;
  params.cpuparams.n_threads = n_threads;
  params.sampling.temp = 0.5;

  model_ = CLI::llava_init(&params);
  if (model_ == NULL) {
      fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
  }
}

LlamaCppParser::~LlamaCppParser() {
  if (!ctx_llava_) {
    ctx_llava_->model = NULL;
    CLI::llava_free(ctx_llava_);
  }
  llama_model_free(model_);
}

struct llava_image_embed * LlamaCppParser::GetEmbedding(std::vector<std::shared_ptr<DNNTensor>> &tensors) {
  llava_image_embed * embed = (llava_image_embed*)malloc(sizeof(llava_image_embed));
  int num_tensors = 0;
  int length_tensors = 0;
  num_tensors = tensors[0]->properties.stride[0] / tensors[0]->properties.stride[1];
  length_tensors = tensors[0]->properties.stride[1] / tensors[0]->properties.stride[2];

  hbUCPMemFlush(&(tensors[0]->sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
  float *image_embed = reinterpret_cast<float *>(tensors[0]->sysMem.virAddr);

  embed->embed = image_embed;
  embed->n_image_pos = num_tensors;
  return embed;
}

int32_t LlamaCppParser::Init(const std::string &system_prompt) {
  ctx_llava_ = CLI::llava_init_context(&params, model_);
  CLI::process_system_prompt(ctx_llava_, &params, system_prompt);
  return 0;
}

int32_t LlamaCppParser::Parse(
                const std::string &user_prompt,
                std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                std::string &result,
                rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher) {

  ggml_time_init();
  params.prompt = user_prompt;

  llava_image_embed * image_embed = GetEmbedding(output_tensors);

  if (!image_embed) {
      return -1;
  }

  // process the prompt
  CLI::process_prompt(ctx_llava_, image_embed, &params, params.prompt, result, publisher);

  llama_perf_context_print(ctx_llava_->ctx_llama);
  free(image_embed);

  ctx_llava_->model = NULL;
  CLI::llava_free(ctx_llava_);

  return 0;
}