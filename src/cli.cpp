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

#include "include/cli.h"

bool CLI::internvl2_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                   int n_batch, int * n_past, int * st_pos_id) {
  int n_embd  = llama_model_n_embd(llama_get_model(ctx_llama));
  const int patch_size = 14 * 2;
  int ph = 16;
  int pw = 16;
  auto img_tokens = image_embed->n_image_pos;
  
  // llama_pos mrope_pos[img_tokens * 4];
  std::vector<llama_pos> mrope_pos;
  mrope_pos.resize(img_tokens * 4);

  for (int y = 0; y < ph; y++)
  {
      for (int x = 0; x < pw; x++)
      {
          int i = y * pw + x;
          mrope_pos[i] = *st_pos_id;
          mrope_pos[i + img_tokens] = *st_pos_id + y;
          mrope_pos[i + img_tokens * 2] = *st_pos_id + x;
          mrope_pos[i + img_tokens * 3] = 0;
      }
  }
  *st_pos_id += std::max(pw, ph);

  int processed = 0;
  std::vector<llama_pos> batch_mrope_pos;
  batch_mrope_pos.resize(img_tokens * 4);

  for (int i = 0; i < img_tokens; i += n_batch) {
      int n_eval = img_tokens - i;
      if (n_eval > n_batch) {
          n_eval = n_batch;
      }

      // llama_pos batch_mrope_pos[n_eval * 4];
      std::fill(batch_mrope_pos.begin(), batch_mrope_pos.end(), 0);
      memcpy(batch_mrope_pos.data(), &mrope_pos[processed], n_eval * sizeof(llama_pos));
      memcpy(&batch_mrope_pos[n_eval * 1], &mrope_pos[img_tokens * 1 + processed], n_eval * sizeof(llama_pos));
      memcpy(&batch_mrope_pos[n_eval * 2], &mrope_pos[img_tokens * 2 + processed], n_eval * sizeof(llama_pos));
      memcpy(&batch_mrope_pos[n_eval * 3], &mrope_pos[img_tokens * 3 + processed], n_eval * sizeof(llama_pos));

      llama_batch batch = {
          int32_t(n_eval),                // n_tokens
          nullptr,                        // token
          (image_embed->embed+i*n_embd),  // embed
          batch_mrope_pos.data(),         // pos
          nullptr,  // n_seq_id
          nullptr,  // seq_id
          nullptr,  // logits
      };

      if (llama_decode(ctx_llama, batch)) {
          LOG_ERR("%s : failed to eval\n", __func__);
          return false;
      }
      *n_past += n_eval;
      processed += n_eval;
  }
  return true;
}

bool CLI::eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past, int * st_pos_id) {
  int N = (int) tokens.size();
  std::vector<llama_pos> pos;
  for (int i = 0; i < N; i += n_batch) {
      int n_eval = (int) tokens.size() - i;
      if (n_eval > n_batch) {
          n_eval = n_batch;
      }
      auto batch = llama_batch_get_one(&tokens[i], n_eval);
      // TODO: add mrope pos ids somewhere else
      pos.resize(batch.n_tokens * 4);
      std::fill(pos.begin(), pos.end(), 0);
      for (int j = 0; j < batch.n_tokens * 3; j ++) {
          pos[j] = *st_pos_id + (j % batch.n_tokens);
      }
      batch.pos = pos.data();

      if (llama_decode(ctx_llama, batch)) {
          LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
          return false;
      }
      *n_past += n_eval;
      *st_pos_id += n_eval;
  }
  return true;
}

bool CLI::eval_id(struct llama_context * ctx_llama, int id, int * n_past, int * st_pos_id) {
  std::vector<llama_token> tokens;
  tokens.push_back(id);
  return eval_tokens(ctx_llama, tokens, 1, n_past, st_pos_id);
}

bool CLI::eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, int * st_pos_id, bool add_bos){
  std::string              str2     = str;
  std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);

  eval_tokens(ctx_llama, embd_inp, n_batch, n_past, st_pos_id);
  return true;
}

const char * CLI::sample(struct common_sampler * smpl,
                         struct llama_context * ctx_llama,
                         int * n_past, int * st_pos_id) {
  const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
  common_sampler_accept(smpl, id, true);

  const llama_model * model = llama_get_model(ctx_llama);
  const llama_vocab * vocab = llama_model_get_vocab(model);

  static std::string ret;
  if (llama_vocab_is_eog(vocab, id)) {
      ret = "</s>";
  } else {
      ret = common_token_to_piece(ctx_llama, id);
  }
  eval_id(ctx_llama, id, n_past, st_pos_id);
  return ret.c_str();
}

void CLI::process_system_prompt(struct llava_context * ctx_llava, common_params * params, const std::string & sprompt) {
  int n_past = 0;
  int cur_pos_id = 0;
  std::string system_prompt = "<|im_start|>system\n" + sprompt + "<|im_end|>\n<|im_start|>user\n<|vision_start|>";
  eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, true);
}

// 判断一个字符是否是中文（UTF-8 三字节范围）
bool isChinese(const std::string& str, size_t i) {
  if (i + 2 >= str.size()) return false;

  unsigned char c1 = str[i];
  unsigned char c2 = str[i + 1];
  unsigned char c3 = str[i + 2];

  // 中文 UTF-8 范围是 E4 B8 80 到 E9 BE A5
  return (c1 >= 0xE4 && c1 <= 0xE9);
}

// 判断是否是中文逗号或句号（UTF-8编码：，=E3 80 81，。=E3 80 82）
bool isChinesePunctuation(const std::string& str, size_t i) {
  if (i + 2 >= str.size()) return false;

  return (str[i] == char(0xE3) && str[i + 1] == char(0x80) &&
          (str[i + 2] == char(0x81) || str[i + 2] == char(0x82)));
}

std::string filterChineseAndPunctuation(const std::string& input, bool& hasChinese, bool& hasPunctuation) {
  std::string result;
  hasChinese = false;
  hasPunctuation = false;

  for (size_t i = 0; i < input.size(); ) {
      if (isChinese(input, i)) {
          hasChinese = true;
          result += input.substr(i, 3);
          i += 3;
      } else {
        hasPunctuation = true;
        if (isChinesePunctuation(input, i)) {
          result += input.substr(i, 3);
          i += 3;
        } else {
          // 英文字符/符号等：跳过
          unsigned char c = input[i];
          if (c < 0x80) {
              ++i;
          } else if ((c & 0xE0) == 0xC0) {
              i += 2; // 2-byte UTF-8
          } else if ((c & 0xF0) == 0xE0) {
              i += 3; // 3-byte UTF-8
          } else if ((c & 0xF8) == 0xF0) {
              i += 4; // 4-byte UTF-8
          } else {
              ++i;
          }
        }
      }
  }

  return result;
}

void CLI::process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, common_params * params, const std::string & prompt, std::string &response, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher) {
  int n_past = 0;
  int cur_pos_id = 0;

  const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

  // llava-1.5 native mode

  if (image_embed != nullptr) {
      internvl2_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past, &cur_pos_id);
  }
  
  std::string user_prompt = "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
  eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, false);

  // generate the response
  LOG("\n");

  struct common_sampler * smpl = common_sampler_init(ctx_llava->model, params->sampling);
  if (!smpl) {
      LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
      exit(1);
  }

  std::string sub_string = "";
  std::vector<std::string> his_strings;
  bool is_repeat = false;
  for (int i = 0; i < max_tgt_len; i++) {
      // 在这里开始融合特征、处理融合特征。
      const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past, &cur_pos_id);

      response += tmp;
      if (strcmp(tmp, "</s>") == 0) break;
      if (strstr(tmp, "###")) break; // Yi-VL behavior
      LOG("%s", tmp);
      if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
      if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
      if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

      bool hasChinese = false;
      bool hasPunctuation = false;
      std::string filtered = filterChineseAndPunctuation(tmp, hasChinese, hasPunctuation);
  
      sub_string += filtered;
      if (hasPunctuation) {
        for (int j = 0; j < his_strings.size(); j++) {
          if (sub_string.size() >= 4 && his_strings[j] == sub_string) {
            is_repeat = true;
            break;
          }
        }

        if (is_repeat) break;
        his_strings.push_back(sub_string);
        std_msgs::msg::String::UniquePtr pub_string(
            new std_msgs::msg::String());  
        pub_string->data = sub_string;
        publisher->publish(std::move(pub_string));
        sub_string = "";
      }
      fflush(stdout);
  }

  common_sampler_free(smpl);
  LOG("\n");
}

struct llama_model * CLI::llava_init(common_params * params) {
  llama_backend_init();
  llama_numa_init(params->numa);

  llama_model_params model_params = common_model_params_to_llama(*params);

  llama_model * model = llama_model_load_from_file(params->model.c_str(), model_params);

  return model;
}

struct llava_context * CLI::llava_init_context(common_params * params, llama_model * model) {
  const char * model_file_name = params->mmproj.c_str();

  auto prompt = params->prompt;
  if (prompt.empty()) {
      prompt = "describe the image in detail.";
  }

  llama_context_params ctx_params = common_context_params_to_llama(*params);
  ctx_params.n_ctx = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

  llama_context * ctx_llama = llama_init_from_model(model, ctx_params);

  if (ctx_llama == NULL) {
      LOG_ERR("%s: failed to create the llama_context\n" , __func__);
      return NULL;
  }

  auto * ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

  ctx_llava->ctx_llama = ctx_llama;
  ctx_llava->model = model;
  return ctx_llava;
}

void CLI::llava_free(struct llava_context * ctx_llava) {
  llama_free(ctx_llava->ctx_llama);
  llama_model_free(ctx_llava->model);
  llama_backend_free();
}
