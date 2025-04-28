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

#include <memory>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

#include "ai_msgs/msg/perception_targets.hpp"
#include "ai_msgs/msg/perf.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "dnn_node/dnn_node_data.h"
#include "dnn_node/util/output_parser/perception_common.h"
#ifdef SHARED_MEM_ENABLED
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#endif

#include "base64.hpp"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"

#include "include/cli.h"
#include "include/post_process/llama_cpp_parser.h"

#ifndef YOLO_WORLD_NODE_H_
#define YOLO_WORLD_NODE_H_

using rclcpp::NodeOptions;

using hobot::dnn_node::DnnNode;
using hobot::dnn_node::DnnNodeOutput;
using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::output_parser::DnnParserResult;
using ai_msgs::msg::PerceptionTargets;

// dnn node输出数据类型
struct ImageEmbeddingOutput : public DnnNodeOutput {

  std::string user_prompt; 

  // 图片数据用于渲染
  std::shared_ptr<hobot::dnn_node::DNNTensor> tensor_image;

  ai_msgs::msg::Perf perf_preprocess;
};

class LlamaCppNode : public DnnNode {
 public:
  LlamaCppNode(const std::string &node_name,
                 const NodeOptions &options = NodeOptions());
  ~LlamaCppNode() override;

 protected:
  // 集成DnnNode的接口，实现参数配置和后处理
  int SetNodePara() override;
  int PostProcess(const std::shared_ptr<DnnNodeOutput> &outputs) override;

 private:
  // 解析配置文件，包好模型文件路径、解析方法等信息
  int LoadVocabulary();

  // 本地回灌进行算法推理
  int FeedFromLocal();

  int Chat();

  int GetTextIndex(
        std::vector<std::string>& user_prompt,
        std::vector<int>& indexs,
        std::vector<std::string>& target_texts);

  // 订阅图片消息的topic和订阅者
  // 共享内存模式
#ifdef SHARED_MEM_ENABLED
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::ConstSharedPtr
      sharedmem_img_subscription_ = nullptr;
  std::string sharedmem_img_topic_name_ = "/hbmem_img";
  void SharedMemImgProcess(
      const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg);
#endif

  // 非共享内存模式
  rclcpp::Subscription<sensor_msgs::msg::Image>::ConstSharedPtr
      ros_img_subscription_ = nullptr;
  // 目前只支持订阅原图，可以使用压缩图"/image_raw/compressed" topic
  // 和sensor_msgs::msg::CompressedImage格式扩展订阅压缩图
  std::string ros_img_sub_topic_name_ = "/image";
  void RosImgProcess(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // string msg 控制话题信息
  rclcpp::Subscription<std_msgs::msg::String>::ConstSharedPtr
      ros_string_subscription_ = nullptr;
  std::string ros_string_sub_topic_name_ = "/prompt_text";
  void RosStringProcess(const std_msgs::msg::String::ConstSharedPtr msg);

  std::shared_ptr<LlamaCppParser> parser_ = nullptr;

  // 用于解析的配置文件，以及解析后的数据
  std::string model_file_name_ = "vit_model_int16_v2.bin";
  std::string model_name_ = "";
  std::string llm_model_name_ = "Qwen2.5-0.5B-Instruct-Q4_0.gguf";

  // 加载模型后，查询出模型输入分辨率
  int model_input_width_ = -1;
  int model_input_height_ = -1;

  // 用于预测的图片来源，0：本地图片；1：订阅到的image msg；2：llamacpp推理
  int feed_type_ = 0;

  // 使用shared mem通信方式订阅图片
  int is_shared_mem_sub_ = 0;

  // 算法推理的任务数
  int task_num_ = 1;
  int llm_threads_ = 8;
  int pre_infer_ = 0;

  // 提示词
  std::string cute_words_ = "好的，让我看看先哈";
  std::string user_prompt_ = "描述一下这张图片.";
  std::string system_prompt_ = "You are a helpful assistant.";
  std::mutex mtx_text_;
  std::mutex mtx_prompt_text_;
  std::string user_true_prompt_;
  std::condition_variable cv_text_;
  std::condition_variable cv_prompt_text_;
  
  bool task_permission_ = true;
  std::mutex mtx_llm_;
  
  std::thread thread_;
  std::atomic<bool> running_;

  // 用于回灌的本地图片信息
  std::string image_file_ = "image2.jpg";

  // 发布AI消息的topic和发布者
  std::string ai_msg_pub_topic_name_ = "/llama_cpp_node";
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr ai_msg_publisher_ =
      nullptr;

  // 发布结果 message
  std::string text_msg_pub_topic_name_ = "/tts_text";
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr output_msg_publisher_ =
      nullptr;
};

#endif  // YOLO_WORLD_NODE_H_
