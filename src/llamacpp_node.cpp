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
#include <unistd.h>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"

#include "include/image_utils.h"
#include "include/llamacpp_node.h"

// 时间格式转换
builtin_interfaces::msg::Time ConvertToRosTime(
    const struct timespec &time_spec) {
  builtin_interfaces::msg::Time stamp;
  stamp.set__sec(time_spec.tv_sec);
  stamp.set__nanosec(time_spec.tv_nsec);
  return stamp;
}

// 根据起始时间计算耗时
int CalTimeMsDuration(const builtin_interfaces::msg::Time &start,
                      const builtin_interfaces::msg::Time &end) {
  return (end.sec - start.sec) * 1000 + end.nanosec / 1000 / 1000 -
         start.nanosec / 1000 / 1000;
}

LlamaCppNode::LlamaCppNode(const std::string &node_name,
                               const NodeOptions &options)
    : DnnNode(node_name, options) {
  // 更新配置
  this->declare_parameter<int>("feed_type", feed_type_);
  this->declare_parameter<std::string>("image", image_file_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<int>("llm_threads", llm_threads_);
  this->declare_parameter<std::string>("llm_model_name", llm_model_name_);
  this->declare_parameter<std::string>("cute_words", cute_words_);
  this->declare_parameter<std::string>("user_prompt", user_prompt_);
  this->declare_parameter<std::string>("system_prompt", system_prompt_);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name_);
  this->declare_parameter<std::string>("text_msg_pub_topic_name",
                                      text_msg_pub_topic_name_);
  this->declare_parameter<std::string>("ros_img_sub_topic_name",
                                       ros_img_sub_topic_name_);
  this->declare_parameter<std::string>("ros_string_sub_topic_name",
                                       ros_string_sub_topic_name_);

  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<std::string>("image", image_file_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<int>("llm_threads", llm_threads_);
  this->get_parameter<std::string>("llm_model_name", llm_model_name_);
  this->get_parameter<std::string>("cute_words", cute_words_);
  this->get_parameter<std::string>("user_prompt", user_prompt_);
  this->get_parameter<std::string>("system_prompt", system_prompt_);
  this->get_parameter<std::string>("ai_msg_pub_topic_name", ai_msg_pub_topic_name_);
  this->get_parameter<std::string>("text_msg_pub_topic_name", text_msg_pub_topic_name_);
  this->get_parameter<std::string>("ros_img_sub_topic_name", ros_img_sub_topic_name_);
  this->get_parameter<std::string>("ros_string_sub_topic_name", ros_string_sub_topic_name_);

  {
    std::stringstream ss;
    ss << "Parameter:"
       << "\n feed_type(0:local, 1:sub): " << feed_type_
       << "\n image: " << image_file_
       << "\n is_shared_mem_sub: " << is_shared_mem_sub_
       << "\n llm_threads: " << llm_threads_
       << "\n llm_model_name: " << llm_model_name_
       << "\n cute_words: " << cute_words_
       << "\n user_prompt: " << user_prompt_
       << "\n system_prompt: " << system_prompt_
       << "\n ai_msg_pub_topic_name: " << ai_msg_pub_topic_name_
       << "\n text_msg_pub_topic_name: " << text_msg_pub_topic_name_
       << "\n ros_img_sub_topic_name: " << ros_img_sub_topic_name_
       << "\n ros_string_sub_topic_name: " << ros_string_sub_topic_name_;
    RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"), "%s", ss.str().c_str());
  }

  // 使用基类接口初始化，加载模型
  if (Init() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Init failed!");
    rclcpp::shutdown();
    return;
  }

  // 未指定模型名，从加载的模型中查询出模型名
  if (model_name_.empty()) {
    if (!GetModel()) {
      RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Get model fail.");
    } else {
      model_name_ = GetModel()->GetName();
      RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"), "Get model name: %s from load model.", model_name_.c_str());
    }
  }

  // 加载模型后查询模型输入分辨率
  if (GetModelInputSize(0, model_input_width_, model_input_height_) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Get model input size fail!");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"),
                "The model input width is %d and height is %d",
                model_input_width_,
                model_input_height_);
  }

  parser_ = std::make_shared<LlamaCppParser>(llm_model_name_, system_prompt_, llm_threads_);

  // 创建AI消息的发布者
  RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"),
              "Create ai msg publisher with topic_name: %s",
              ai_msg_pub_topic_name_.c_str());
  ai_msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
      ai_msg_pub_topic_name_, 10);

  output_msg_publisher_ = this->create_publisher<std_msgs::msg::String>(
    text_msg_pub_topic_name_, 10);

  if (0 == feed_type_) {
    // 本地图片回灌
    RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"),
                "Dnn node feed with local image: %s",
                image_file_.c_str());
    FeedFromLocal();
  } 
  else if (1 == feed_type_) {
    // 创建图片消息的订阅者
    RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"),
                "Dnn node feed with subscription");
    
    RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"),
      "Create string subscription with topic_name: %s",
      ros_string_sub_topic_name_.c_str());
    ros_string_subscription_ =
          this->create_subscription<std_msgs::msg::String>(
              ros_string_sub_topic_name_,
              10,
              std::bind(
                  &LlamaCppNode::RosStringProcess, this, std::placeholders::_1));
    if (is_shared_mem_sub_) {
#ifdef SHARED_MEM_ENABLED
      RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"),
                  "Create img hbmem_subscription with topic_name: %s",
                  sharedmem_img_topic_name_.c_str());
      sharedmem_img_subscription_ =
          this->create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              sharedmem_img_topic_name_,
              rclcpp::SensorDataQoS(),
              std::bind(&LlamaCppNode::SharedMemImgProcess,
                        this,
                        std::placeholders::_1));
#else
      RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Unsupport shared mem");
#endif
    } else {
      RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"),
                  "Create img subscription with topic_name: %s",
                  ros_img_sub_topic_name_.c_str());
      ros_img_subscription_ =
          this->create_subscription<sensor_msgs::msg::Image>(
              ros_img_sub_topic_name_,
              10,
              std::bind(
                  &LlamaCppNode::RosImgProcess, this, std::placeholders::_1));
    }
  } else {
    RCLCPP_ERROR(
        rclcpp::get_logger("llama_cpp_node"), "Invalid feed_type:%d", feed_type_);
    rclcpp::shutdown();
    return;
  }
}

LlamaCppNode::~LlamaCppNode() {
  {
    std::unique_lock<std::mutex> lg(mtx_text_);
    cv_text_.notify_all();
    lg.unlock();
  }
  {
    std::unique_lock<std::mutex> lg(mtx_llm_);
    lg.unlock();
  }
}

int LlamaCppNode::SetNodePara() {
  RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"), "Set node para.");
  if (!dnn_node_para_ptr_) {
    return -1;
  }
  dnn_node_para_ptr_->model_file = model_file_name_;
  dnn_node_para_ptr_->model_name = model_name_;
  dnn_node_para_ptr_->model_task_type =
      hobot::dnn_node::ModelTaskType::ModelInferType;
  dnn_node_para_ptr_->task_num = task_num_;

  RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"),
              "model_file_name: %s, task_num: %d",
              model_file_name_.data(),
              dnn_node_para_ptr_->task_num);

  return 0;
}

int LlamaCppNode::GetTextIndex(
      std::vector<std::string>& user_prompt,
      std::vector<int>& indexs,
      std::vector<std::string>& target_texts) {
  return 0;
}

int LlamaCppNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput> &node_output) {
  if (!rclcpp::ok()) {
    return -1;
  }

  std_msgs::msg::String::UniquePtr pub_string(
      new std_msgs::msg::String());
  pub_string->data = cute_words_;
  output_msg_publisher_->publish(std::move(pub_string));

  // 1. 记录后处理开始时间
  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

  auto parser_output = std::dynamic_pointer_cast<ImageEmbeddingOutput>(node_output);
  if (parser_output) {
    std::stringstream ss;
    ss << "Output from frame_id: " << parser_output->msg_header->frame_id
       << ", stamp: " << parser_output->msg_header->stamp.sec << "."
       << parser_output->msg_header->stamp.nanosec;
    RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"), "%s", ss.str().c_str());
  }

  // 校验算法输出是否有效
  if (node_output->output_tensors.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"),
                 "Invalid node_output->output_tensors");
    return -1;
  }

  // 2. 模型后处理解析
  std::string result = "";
  parser_->Init(system_prompt_);
  parser_->Parse(parser_output->user_prompt, parser_output->output_tensors, result, output_msg_publisher_);
  if (parser_output) {
    std::stringstream ss;
    ss << result;
    RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"), "\n%s", ss.str().c_str());
  }

  {
    std::unique_lock<std::mutex> lg(mtx_llm_);
    task_permission_ = true;
    lg.unlock();
  }

  // 3. 创建用于发布的AI消息
  if (!ai_msg_publisher_) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Invalid msg_publisher");
    return -1;
  }
  ai_msgs::msg::PerceptionTargets::UniquePtr pub_data(
      new ai_msgs::msg::PerceptionTargets());
  // 3.1 发布检测AI消息
  ai_msgs::msg::Target target;
  target.set__type(result);
  pub_data->targets.emplace_back(std::move(target));

  pub_data->header.set__stamp(parser_output->msg_header->stamp);
  pub_data->header.set__frame_id(parser_output->msg_header->frame_id);

  // 填充perf性能统计信息
  // 前处理统计
  ai_msgs::msg::Perf perf_preprocess = std::move(parser_output->perf_preprocess);
  perf_preprocess.set__time_ms_duration(CalTimeMsDuration(
      perf_preprocess.stamp_start, perf_preprocess.stamp_end));

  // dnn node有输出统计信息
  if (node_output->rt_stat) {
    struct timespec time_now = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_now);

    // 推理统计
    ai_msgs::msg::Perf perf;
    perf.set__type(model_name_ + "_predict_infer");
    perf.stamp_start =
        ConvertToRosTime(node_output->rt_stat->infer_timespec_start);
    perf.stamp_end = ConvertToRosTime(node_output->rt_stat->infer_timespec_end);
    perf.set__time_ms_duration(node_output->rt_stat->infer_time_ms);
    pub_data->perfs.push_back(perf);

    perf.set__type(model_name_ + "_predict_parse");
    perf.stamp_start =
        ConvertToRosTime(node_output->rt_stat->parse_timespec_start);
    perf.stamp_end = ConvertToRosTime(node_output->rt_stat->parse_timespec_end);
    perf.set__time_ms_duration(node_output->rt_stat->parse_time_ms);
    pub_data->perfs.push_back(perf);

    // 后处理统计
    ai_msgs::msg::Perf perf_postprocess;
    perf_postprocess.set__type(model_name_ + "_postprocess");
    perf_postprocess.stamp_start = ConvertToRosTime(time_start);
    clock_gettime(CLOCK_REALTIME, &time_now);
    perf_postprocess.stamp_end = ConvertToRosTime(time_now);
    perf_postprocess.set__time_ms_duration(CalTimeMsDuration(
        perf_postprocess.stamp_start, perf_postprocess.stamp_end));
    pub_data->perfs.emplace_back(perf_postprocess);

    // 推理输出帧率统计
    pub_data->set__fps(round(node_output->rt_stat->output_fps));

    // 如果当前帧有更新统计信息，输出统计信息
    if (node_output->rt_stat->fps_updated) {
      RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"),
                  "Sub img fps: %.2f, Smart fps: %.2f, pre process time ms: %d, "
                  "infer time ms: %d, "
                  "post process time ms: %d",
                  node_output->rt_stat->input_fps,
                  node_output->rt_stat->output_fps,
                  static_cast<int>(perf_preprocess.time_ms_duration),
                  node_output->rt_stat->infer_time_ms,
                  static_cast<int>(perf_postprocess.time_ms_duration));
    }
  }

  // 发布AI消息
  ai_msg_publisher_->publish(std::move(pub_data));
  return 0;
}

int LlamaCppNode::FeedFromLocal() {
  if (access(image_file_.c_str(), R_OK) == -1) {
    RCLCPP_ERROR(
        rclcpp::get_logger("llama_cpp_node"), "Image: %s not exist!", image_file_.c_str());
    return -1;
  }

  auto dnn_output = std::make_shared<ImageEmbeddingOutput>();
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_start.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_now.tv_nsec;

  // 1. 获取图片数据DNNTensor
  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 0);
  std::shared_ptr<DNNTensor> tensor_image = nullptr;

  cv::Mat bgr_mat = cv::imread(image_file_, cv::IMREAD_COLOR);
  tensor_image = ImageUtils::GetBGRTensorFromBGR(bgr_mat,
      model_input_height_, model_input_width_, tensor_properties);

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("ClipImageNode"),
                 "Get tensor fail with image: %s",
                 image_file_.c_str());
    return -1;
  }

  // 2. 存储上面DNNTensor
  // inputs将会作为模型的输入通过InferTask接口传入
  std::vector<std::shared_ptr<DNNTensor>> inputs;
  inputs.push_back(tensor_image);
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id("feedback");
  dnn_output->user_prompt = user_prompt_;

  // 3. 开始预测
  if (Run(inputs, dnn_output, true) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Run predict failed!");
    return -1;
  }

  return 0;
}

void LlamaCppNode::RosImgProcess(
    const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
  if (!img_msg) {
    RCLCPP_DEBUG(rclcpp::get_logger("llama_cpp_node"), "Get img failed");
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  std::stringstream ss;
  ss << "Recved img encoding: " << img_msg->encoding
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step
     << ", frame_id: " << img_msg->header.frame_id
     << ", stamp: " << img_msg->header.stamp.sec << "_"
     << img_msg->header.stamp.nanosec
     << ", data size: " << img_msg->data.size();
  RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"), "%s", ss.str().c_str());

  auto dnn_output = std::make_shared<ImageEmbeddingOutput>();
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_start.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_now.tv_nsec;

  // 1. 将图片处理成模型输入数据类型DNNTensor
  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 0);
  std::shared_ptr<DNNTensor> tensor_image = nullptr;
  if ("rgb8" == img_msg->encoding) {
    auto cv_img =
        cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(img_msg), "bgr8");
    tensor_image = ImageUtils::GetBGRTensorFromBGR(cv_img->image,
          model_input_height_, model_input_width_, tensor_properties);
  } else if ("bgr8" == img_msg->encoding) {
    auto cv_img =
        cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(img_msg), "bgr8");
    tensor_image = ImageUtils::GetBGRTensorFromBGR(cv_img->image,
          model_input_height_, model_input_width_, tensor_properties);
  } else if ("nv12" == img_msg->encoding) {  // nv12格式使用hobotcv resize
    cv::Mat bgr_mat;
    hobot::dnn_node::ImageProc::Nv12ToBGR(reinterpret_cast<const char *>(img_msg->data.data()), img_msg->height, img_msg->width, bgr_mat);
    tensor_image = ImageUtils::GetBGRTensorFromBGR(bgr_mat,
          model_input_height_, model_input_width_, tensor_properties);
  }

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Get Tensor fail");
    return;
  }

  // 2. 存储上面两个DNNTensor
  // inputs将会作为模型的输入通过InferTask接口传入
  auto inputs = std::vector<std::shared_ptr<DNNTensor>>{tensor_image};
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(img_msg->header.frame_id);
  dnn_output->msg_header->set__stamp(img_msg->header.stamp);
  
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  dnn_output->user_prompt = user_prompt_;

  // 3. 开始预测
  int ret = Run(inputs, dnn_output, false);
  if (ret != 0 && ret != HB_DNN_TASK_NUM_EXCEED_LIMIT) {
    RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"), "Run predict failed!");
    return;
  }
}

#ifdef SHARED_MEM_ENABLED
void LlamaCppNode::SharedMemImgProcess(
    const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr img_msg) {
  if (!img_msg) {
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  std::stringstream ss;
  ss << "Recved img encoding: "
     << std::string(reinterpret_cast<const char *>(img_msg->encoding.data()))
     << ", h: " << img_msg->height << ", w: " << img_msg->width
     << ", step: " << img_msg->step << ", index: " << img_msg->index
     << ", stamp: " << img_msg->time_stamp.sec << "_"
     << img_msg->time_stamp.nanosec << ", data size: " << img_msg->data_size;
  RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"), "%s", ss.str().c_str());

  auto dnn_output = std::make_shared<ImageEmbeddingOutput>();
  struct timespec time_now = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_start.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_start.nanosec = time_now.tv_nsec;

  {
    std::unique_lock<std::mutex> lg(mtx_text_);
    if (user_prompt_ == "") {
      return;
    }
    dnn_output->user_prompt = user_prompt_;
    user_prompt_ = "";
  }
  {
    std::unique_lock<std::mutex> lg(mtx_llm_);
    if (!task_permission_) {
      return;  // 直接返回，不等
    }
    task_permission_ = false;  // 占用
  }

  // 1. 将图片处理成模型输入数据类型DNNTensor
  auto model = GetModel();
  hbDNNTensorProperties tensor_properties;
  model->GetInputTensorProperties(tensor_properties, 0);
  std::shared_ptr<DNNTensor> tensor_image = nullptr;
  if ("nv12" ==
      std::string(reinterpret_cast<const char *>(img_msg->encoding.data()))) {
    cv::Mat bgr_mat;
    hobot::dnn_node::ImageProc::Nv12ToBGR(reinterpret_cast<const char *>(img_msg->data.data()), img_msg->height, img_msg->width, bgr_mat);
    tensor_image = ImageUtils::GetBGRTensorFromBGR(bgr_mat,
                    model_input_height_, model_input_width_, tensor_properties);
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"),
                 "Unsupported img encoding: %s, only nv12 img encoding is "
                 "supported for shared mem.",
                 img_msg->encoding.data());
    return;
  }

  if (!tensor_image) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Get Tensor fail");
    return;
  }

  {
    auto stamp_start = ConvertToRosTime(time_now);
    struct timespec time_end = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time_end);
    auto stamp_end = ConvertToRosTime(time_end);
    RCLCPP_DEBUG(rclcpp::get_logger("llama_cpp_node"),
            "image preforcess time: %d", 
            CalTimeMsDuration(stamp_start, stamp_end));
  }

  // 2. 初始化输出
  auto inputs = std::vector<std::shared_ptr<DNNTensor>>{tensor_image};
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->msg_header->set__stamp(img_msg->time_stamp);
  
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");

  // 3. 开始预测
  int ret = Run(inputs, dnn_output, false);
  if (ret != 0 && ret != HB_DNN_TASK_NUM_EXCEED_LIMIT) {
    RCLCPP_ERROR(rclcpp::get_logger("llama_cpp_node"), "Run predict failed!");
    return;
  }
}
#endif

void LlamaCppNode::RosStringProcess(
    const std_msgs::msg::String::ConstSharedPtr msg) {
  if (!msg) {
    RCLCPP_DEBUG(rclcpp::get_logger("llama_cpp_node"), "Get string failed");
    return;
  }

  if (!rclcpp::ok()) {
    return;
  }

  std::stringstream ss;
  ss << "Recved string data: " << msg->data;
  RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"), "%s", ss.str().c_str());

  std::unique_lock<std::mutex> lg(mtx_text_);
  user_prompt_ = msg->data;
  cv_text_.notify_one();
  lg.unlock();
}
