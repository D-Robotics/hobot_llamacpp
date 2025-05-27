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
#include <random>
#include <regex>

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
  this->declare_parameter<int>("parser_mode", parser_mode_);
  this->declare_parameter<std::string>("image", image_file_);
  this->declare_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->declare_parameter<int>("llm_threads", llm_threads_);
  this->declare_parameter<std::string>("llm_model_name", llm_model_name_);
  this->declare_parameter<std::string>("model_file_name", model_file_name_);
  this->declare_parameter<std::string>("cute_words", cute_words_);
  this->declare_parameter<std::string>("user_prompt", user_prompt_);
  this->declare_parameter<std::string>("system_prompt", system_prompt_);
  this->declare_parameter<int>("pre_infer", pre_infer_);
  this->declare_parameter<std::string>("ai_msg_pub_topic_name",
                                       ai_msg_pub_topic_name_);
  this->declare_parameter<std::string>("text_msg_pub_topic_name",
                                      text_msg_pub_topic_name_);
  this->declare_parameter<std::string>("ros_img_sub_topic_name",
                                       ros_img_sub_topic_name_);
  this->declare_parameter<std::string>("ros_string_sub_topic_name",
                                       ros_string_sub_topic_name_);

  this->get_parameter<int>("feed_type", feed_type_);
  this->get_parameter<int>("parser_mode", parser_mode_);
  this->get_parameter<std::string>("image", image_file_);
  this->get_parameter<int>("is_shared_mem_sub", is_shared_mem_sub_);
  this->get_parameter<int>("llm_threads", llm_threads_);
  this->get_parameter<std::string>("llm_model_name", llm_model_name_);
  this->get_parameter<std::string>("model_file_name", model_file_name_);
  this->get_parameter<std::string>("cute_words", cute_words_);
  this->get_parameter<std::string>("user_prompt", user_prompt_);
  this->get_parameter<std::string>("system_prompt", system_prompt_);
  this->get_parameter<int>("pre_infer", pre_infer_);
  this->get_parameter<std::string>("ai_msg_pub_topic_name", ai_msg_pub_topic_name_);
  this->get_parameter<std::string>("text_msg_pub_topic_name", text_msg_pub_topic_name_);
  this->get_parameter<std::string>("ros_img_sub_topic_name", ros_img_sub_topic_name_);
  this->get_parameter<std::string>("ros_string_sub_topic_name", ros_string_sub_topic_name_);

  {
    std::stringstream ss;
    ss << "Parameter:"
       << "\n feed_type(0:local, 1:sub): " << feed_type_
       << "\n parser_mode(0:common, 1:det): " << parser_mode_
       << "\n image: " << image_file_
       << "\n is_shared_mem_sub: " << is_shared_mem_sub_
       << "\n llm_threads: " << llm_threads_
       << "\n llm_model_name: " << llm_model_name_
       << "\n model_file_name: " << model_file_name_
       << "\n cute_words: " << cute_words_
       << "\n user_prompt: " << user_prompt_
       << "\n system_prompt: " << system_prompt_
       << "\n pre_infer: " << pre_infer_
       << "\n ai_msg_pub_topic_name: " << ai_msg_pub_topic_name_
       << "\n text_msg_pub_topic_name: " << text_msg_pub_topic_name_
       << "\n ros_img_sub_topic_name: " << ros_img_sub_topic_name_
       << "\n ros_string_sub_topic_name: " << ros_string_sub_topic_name_;
    RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"), "%s", ss.str().c_str());
  }

  if (feed_type_ == 0 || feed_type_ == 1) {
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

    parser_ = std::make_shared<LlamaCppParser>(llm_model_name_, system_prompt_, llm_threads_);
  }
  
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
    pre_infer_ = 0;
    FeedFromLocal();
  } 
  else if (1 == feed_type_ || 2 == feed_type_) {
    RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"),
    "Create string subscription with topic_name: %s",
    ros_string_sub_topic_name_.c_str());
    ros_string_subscription_ =
        this->create_subscription<std_msgs::msg::String>(
            ros_string_sub_topic_name_,
            10,
            std::bind(
                &LlamaCppNode::RosStringProcess, this, std::placeholders::_1));
    if (1 == feed_type_) {
      // 创建图片消息的订阅者
      RCLCPP_INFO(rclcpp::get_logger("llama_cpp_node"),
                  "Dnn node feed with subscription");
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
    } else if (2 == feed_type_) {
      thread_ = std::thread(&LlamaCppNode::Chat, this);
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
    std::unique_lock<std::mutex> lg(mtx_prompt_text_);
    cv_prompt_text_.notify_all();
    lg.unlock();
  }
  {
    std::unique_lock<std::mutex> lg(mtx_llm_);
    lg.unlock();
  }
  running_ = false;
  if (thread_.joinable())
  {
    thread_.join();
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

int LlamaCppNode::PostProcess(
    const std::shared_ptr<DnnNodeOutput> &node_output) {
  if (!rclcpp::ok()) {
    return -1;
  }
  auto parser_output = std::dynamic_pointer_cast<ImageEmbeddingOutput>(node_output);
  if (pre_infer_ == 1){
    std::unique_lock<std::mutex> lg(mtx_prompt_text_);
    cv_prompt_text_.wait(lg, [this]() { return user_true_prompt_ != "" || !rclcpp::ok(); });
    parser_output->user_prompt = user_true_prompt_;
    user_true_prompt_ = "";
    lg.unlock();
  }

  std_msgs::msg::String::UniquePtr pub_string(
      new std_msgs::msg::String());
  pub_string->data = getRandomSubword(cute_words_);
  output_msg_publisher_->publish(std::move(pub_string));

  // 1. 记录后处理开始时间
  struct timespec time_start = {0, 0};
  clock_gettime(CLOCK_REALTIME, &time_start);

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
  if (parser_mode_ == 0) {
    ai_msgs::msg::Target target;
    target.set__type(result);
    pub_data->targets.emplace_back(std::move(target));
  } else if (parser_mode_ == 1) {
  
    auto [class_name, xmin, ymin, xmax, ymax] = parse_detection_string(result);
    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax >= 1000) xmax = 1000 - 1;
    if (ymax >= 1000) ymax = 1000 - 1;

    if (xmax != 0 && ymax != 0) {

      xmin = static_cast<int>(static_cast<float>(xmin)  * static_cast<float>(parser_output->img_w) / 1000.0);
      ymin = static_cast<int>(static_cast<float>(ymin)  * static_cast<float>(parser_output->img_w) / 1000.0);
      xmax = static_cast<int>(static_cast<float>(xmax)  * static_cast<float>(parser_output->img_w) / 1000.0);
      ymax = static_cast<int>(static_cast<float>(ymax)  * static_cast<float>(parser_output->img_w) / 1000.0);

      std::stringstream ss;
      ss << "det rect: " << xmin << " " << ymin << " "
          << xmax << " " << ymax
          << ", det type: " << class_name;
      RCLCPP_WARN(rclcpp::get_logger("llama_cpp_node"), "%s", ss.str().c_str());

      ai_msgs::msg::Roi roi;
      roi.set__type(class_name);
      roi.rect.set__x_offset(xmin);
      roi.rect.set__y_offset(ymin);
      roi.rect.set__width(xmax - xmin);
      roi.rect.set__height(ymax - ymin);
      roi.set__confidence(1.0);

      ai_msgs::msg::Target target;
      target.set__type(class_name);
      target.rois.emplace_back(roi);
      pub_data->targets.emplace_back(std::move(target)); 
    }
  }

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
  if (parser_mode_ == 1) {
    dnn_output->user_prompt = "Please provide the bounding box coordinate of the region this sentence describes: <ref>" + user_prompt_ + "</ref>";
  }

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

  {
    std::unique_lock<std::mutex> lg(mtx_text_);
    if (user_prompt_ == "") {
      return;
    }
    dnn_output->user_prompt = user_prompt_;
    if (parser_mode_ == 1) {
      dnn_output->user_prompt = "Please provide the bounding box coordinate of the region this sentence describes: <ref>" + user_prompt_ + "</ref>";
    }
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
  dnn_output->img_w = (img_msg->width > img_msg->height) ? img_msg->width : img_msg->height;
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(img_msg->header.frame_id);
  dnn_output->msg_header->set__stamp(img_msg->header.stamp);
  
  clock_gettime(CLOCK_REALTIME, &time_now);
  dnn_output->perf_preprocess.stamp_end.sec = time_now.tv_sec;
  dnn_output->perf_preprocess.stamp_end.nanosec = time_now.tv_nsec;
  dnn_output->perf_preprocess.set__type(model_name_ + "_preprocess");
  dnn_output->user_prompt = user_prompt_;
  if (parser_mode_ == 1) {
    dnn_output->user_prompt = "Please provide the bounding box coordinate of the region this sentence describes: <ref>" + user_prompt_ + "</ref>";
  }

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
    if (parser_mode_ == 1) {
      dnn_output->user_prompt = "Please provide the bounding box coordinate of the region this sentence describes: <ref>" + user_prompt_ + "</ref>";
    }
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
  dnn_output->img_w = (img_msg->width > img_msg->height) ? img_msg->width : img_msg->height;
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

  if (pre_infer_ == 1) {
    if (msg->data == "你好" || msg->data == "你好," || msg->data == "你好，") {
      {
        std::unique_lock<std::mutex> lg(mtx_text_);
        user_prompt_ = msg->data;
        cv_text_.notify_one();
        lg.unlock();
      }
      {
        std::unique_lock<std::mutex> lg(mtx_prompt_text_);
        user_true_prompt_ = "";
        lg.unlock();
      }
      return;
    } else {  
      std::unique_lock<std::mutex> lg(mtx_prompt_text_);
      user_true_prompt_ = msg->data;
      cv_prompt_text_.notify_one();
      lg.unlock();
      return;
    }
  }

  std::unique_lock<std::mutex> lg(mtx_text_);
  user_prompt_ = msg->data;
  cv_text_.notify_one();
  lg.unlock();
}


int LlamaCppNode::Chat() {
  common_params params;
  g_params = &params;
  int argc;
  char ** argv;
  if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
      return 1;
  }
  params.model = llm_model_name_;
  params.cpuparams.n_threads = llm_threads_;
  params.sampling.temp = 0.5;
  params.n_predict = 128;
  std::string value = system_prompt_;
  std::ifstream file(value);
  if (!file) {
      throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
  }
  // store the external file name in params
  params.prompt_file = value;
  std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
  if (!params.prompt.empty() && params.prompt.back() == '\n') {
      params.prompt.pop_back();
  }

  common_init();

  auto & sparams = params.sampling;

  // save choice to use color for later
  // (note for later: this is a slightly awkward choice)
  console::init(params.simple_io, params.use_color);
  atexit([]() { console::cleanup(); });

  llama_backend_init();
  llama_numa_init(params.numa);

  llama_model * model = nullptr;
  llama_context * ctx = nullptr;
  common_sampler * smpl = nullptr;

  g_model = &model;
  g_ctx = &ctx;
  g_smpl = &smpl;

  std::vector<common_chat_msg> chat_msgs;

  // load the model and apply lora adapter, if any
  LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
  common_init_result llama_init = common_init_from_params(params);

  model = llama_init.model.get();
  ctx = llama_init.context.get();

  if (model == NULL) {
      LOG_ERR("%s: error: unable to load model\n", __func__);
      return 1;
  }

  const llama_vocab * vocab = llama_model_get_vocab(model);
  auto chat_templates = common_chat_templates_init(model, params.chat_template);

  LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

  auto * reg = ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
  auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
  auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

  struct ggml_threadpool_params tpp_batch =
          ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
  struct ggml_threadpool_params tpp =
          ggml_threadpool_params_from_cpu_params(params.cpuparams);

  set_process_priority(params.cpuparams.priority);

  struct ggml_threadpool * threadpool_batch = NULL;
  if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
      threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
      if (!threadpool_batch) {
          LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
          return 1;
      }

      // Start the non-batch threadpool in the paused state
      tpp.paused = true;
  }

  struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
  if (!threadpool) {
      LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
      return 1;
  }

  llama_attach_threadpool(ctx, threadpool, threadpool_batch);

  const int n_ctx_train = llama_model_n_ctx_train(model);
  const int n_ctx = llama_n_ctx(ctx);

  if (n_ctx > n_ctx_train) {
      LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
  }

  // auto enable conversation mode if chat template is available
  const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
  if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
      if (has_chat_template) {
          LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
          params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
      } else {
          params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
      }
  }

  // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
  if (params.conversation_mode && !has_chat_template) {
      LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
  }

  // print chat template example in conversation mode
  if (params.conversation_mode) {
      if (params.enable_chat_template) {
          LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja).c_str());
      } else {
          LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
      }
  }

  // print system information
  {
      LOG_INF("\n");
      LOG_INF("%s\n", common_params_get_system_info(params).c_str());
      LOG_INF("\n");
  }

  std::string path_session = params.path_prompt_cache;
  std::vector<llama_token> session_tokens;

  const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
  if (!llama_model_has_encoder(model)) {
      GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
  }

  LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

  std::vector<llama_token> embd_inp;

  auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
      common_chat_msg new_msg;
      new_msg.role = role;
      new_msg.content = content;
      auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
      chat_msgs.push_back(new_msg);
      LOG_DBG("formatted: '%s'\n", formatted.c_str());
      return formatted;
  };

  {
      auto prompt = (params.conversation_mode && params.enable_chat_template)
          // format the system prompt in conversation mode (fallback to default if empty)
          ? chat_add_and_format("system", params.prompt.empty() ? DEFAULT_SYSTEM_MESSAGE : params.prompt)
          // otherwise use the prompt as is
          : params.prompt;
      embd_inp = common_tokenize(ctx, prompt, true, true);
      LOG_INF("prompt: \"%s\"\n", prompt.c_str());
      LOG_INF("tokens: %s\n", string_from(ctx, embd_inp).c_str());
  }

  // Should not run without any tokens
  if (embd_inp.empty()) {
      if (add_bos) {
          embd_inp.push_back(llama_vocab_bos(vocab));
          LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
      } else {
          LOG_ERR("input is empty\n");
          return -1;
      }
  }

  // Tokenize negative prompt
  if ((int) embd_inp.size() > n_ctx - 4) {
      LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
      return 1;
  }

  // debug message about similarity of saved session, if applicable
  size_t n_matching_session_tokens = 0;

  LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
       embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

  // if we will use the cache for the full prompt without reaching the end of the cache, force
  // reevaluation of the last token to recalculate the cached logits
  if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
      LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", embd_inp.size() - 1);

      session_tokens.resize(embd_inp.size() - 1);
  }

  // number of tokens to keep when resetting context
  if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
      params.n_keep = (int)embd_inp.size();
  } else {
      params.n_keep += add_bos; // always keep the BOS token
  }

  if (params.conversation_mode) {
      params.interactive_first = true;
  }

  // enable interactive mode if interactive start is specified
  if (params.interactive_first) {
      params.interactive = true;
  }

  // params.interactive = false;
  if (params.interactive) {
      LOG_INF("%s: interactive mode on.\n", __func__);
  }

  smpl = common_sampler_init(model, sparams);
  if (!smpl) {
      LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
      return 1;
  }

  LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
  LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
  LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());
  LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

  // group-attention state
  // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
  int ga_i = 0;

  const int ga_n = params.grp_attn_n;
  const int ga_w = params.grp_attn_w;
  LOG_INF("\n");
  is_interacting = params.interactive_first;

  bool is_antiprompt        = false;
  bool input_echo           = true;
  bool display              = true;
  bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

  int n_past             = 0;
  int n_remain           = params.n_predict;
  int n_consumed         = 0;
  int n_session_consumed = 0;

  std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
  std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
  std::ostringstream output_ss;     g_output_ss     = &output_ss;
  std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

  // the first thing we will do is to output the prompt, so set color accordingly
  console::set_display(console::prompt);
  display = params.display_prompt;

  std::vector<llama_token> embd;

  // single-token antiprompts
  std::vector<llama_token> antiprompt_token;

  for (const std::string & antiprompt : params.antiprompt) {
      auto ids = ::common_tokenize(ctx, antiprompt, false, true);
      if (ids.size() == 1) {
          antiprompt_token.push_back(ids[0]);
      }
  }

  if (llama_model_has_encoder(model)) {
      int enc_input_size = embd_inp.size();
      llama_token * enc_input_buf = embd_inp.data();

      if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
          LOG_ERR("%s : failed to eval\n", __func__);
          return 1;
      }

      llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
      if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
          decoder_start_token_id = llama_vocab_bos(vocab);
      }

      embd_inp.clear();
      embd_inp.push_back(decoder_start_token_id);
  }

  bool start = true;
  running_ = true;
  std::string sub_string = "";
  std::string result_string = "";
  std::vector<std::string> his_strings;
  bool is_repeat = false;
  bool init_status = false;
  while (rclcpp::ok() && running_ && ((n_remain != 0 && !is_antiprompt) || params.interactive)) {
      // predict
      if (!embd.empty()) {
          // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
          // --prompt or --file which uses the same value.
          int max_embd_size = n_ctx - 4;

          if (ga_n == 1) {
              // infinite text generation via context shifting
              // if we run out of context:
              // - take the n_keep first tokens from the original prompt (via n_past)
              // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

              if (n_past + (int) embd.size() >= n_ctx) {
                  if (!params.ctx_shift){
                      LOG_DBG("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                      break;
                  }

                  if (params.n_predict == -2) {
                      LOG_DBG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                      break;
                  }

                  const int n_left    = n_past - params.n_keep;
                  const int n_discard = n_left/2;

                  LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                          n_past, n_left, n_ctx, params.n_keep, n_discard);

                  llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
                  llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                  n_past -= n_discard;

                  LOG_DBG("after swap: n_past = %d\n", n_past);

                  LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                  LOG_DBG("clear session path\n");
                  path_session.clear();
              }
          } 

          // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
          if (n_session_consumed < (int) session_tokens.size()) {
              size_t i = 0;
              for ( ; i < embd.size(); i++) {
                  if (embd[i] != session_tokens[n_session_consumed]) {
                      session_tokens.resize(n_session_consumed);
                      break;
                  }

                  n_past++;
                  n_session_consumed++;

                  if (n_session_consumed >= (int) session_tokens.size()) {
                      ++i;
                      break;
                  }
              }
              if (i > 0) {
                  embd.erase(embd.begin(), embd.begin() + i);
              }
          }

          for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
              int n_eval = (int) embd.size() - i;
              if (n_eval > params.n_batch) {
                  n_eval = params.n_batch;
              }

              LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

              if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                  LOG_ERR("%s : failed to eval\n", __func__);
                  return 1;
              }

              n_past += n_eval;

              LOG_DBG("n_past = %d\n", n_past);
              // Display total tokens alongside total time
              if (params.n_print > 0 && n_past % params.n_print == 0) {
                  LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
              }
          }

          if (!embd.empty() && !path_session.empty()) {
              session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
              n_session_consumed = session_tokens.size();
          }
      }

      embd.clear();

      if ((int) embd_inp.size() <= n_consumed && !is_interacting) {

          // optionally save the session on first sample (for faster prompt loading next time)
          if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
              need_to_save_session = false;
              llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

              LOG_DBG("saved session to %s\n", path_session.c_str());
          }

          const llama_token id = common_sampler_sample(smpl, ctx, -1);

          common_sampler_accept(smpl, id, /* accept_grammar= */ true);

          // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

          embd.push_back(id);

          // echo this to console
          input_echo = true;

          // decrement remaining sampling budget
          --n_remain;

          LOG_DBG("n_remain: %d\n", n_remain);
      } else {
          // some user input remains from prompt or interaction, forward it to processing
          LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
          while ((int) embd_inp.size() > n_consumed) {
              embd.push_back(embd_inp[n_consumed]);

              // push the prompt in the sampling context in order to apply repetition penalties later
              // for the prompt, we don't apply grammar rules
              common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

              ++n_consumed;
              if ((int) embd.size() >= params.n_batch) {
                  break;
              }
          }
      }

      // display text
      if (input_echo && display && !task_permission_) {
          for (auto id : embd) {
              const std::string token_str = common_token_to_piece(ctx, id, params.special);
              LOG("%s", token_str.c_str());
              // Console/Stream Output

              bool hasChinese = false;
              bool hasPunctuation = false;
              std::string filtered = filterChineseAndPunctuation(token_str, hasChinese, hasPunctuation);

              sub_string += filtered;
              result_string += token_str;
              if (hasPunctuation) {
                for (int j = 0; j < his_strings.size(); j++) {
                  // if (sub_string.size() >= 4 && his_strings[j] == sub_string) {
                  //   is_repeat = true;
                  //   break;
                  // }
                }
                if (sub_string == "") continue;
                // if (is_repeat) break;
                his_strings.push_back(sub_string);
                std_msgs::msg::String::UniquePtr pub_string(
                    new std_msgs::msg::String());  
                pub_string->data = sub_string;
                output_msg_publisher_->publish(std::move(pub_string));
                sub_string = "";
              }
              // Record Displayed Tokens To Log
              // Note: Generated tokens are created one by one hence this check
              if (embd.size() > 1) {
                  // Incoming Requested Tokens
                  input_tokens.push_back(id);
              } else {
                  // Outgoing Generated Tokens
                  output_tokens.push_back(id);
                  output_ss << token_str;
              }
          }
      }

      // reset color to default if there is no pending user input
      if (input_echo && (int) embd_inp.size() == n_consumed) {
          console::set_display(console::reset);
          display = true;
      }

      // if not currently processing queued inputs;
      if ((int) embd_inp.size() <= n_consumed) {
          // check for reverse prompt in the last n_prev tokens
          if (!params.antiprompt.empty()) {
              const int n_prev = 32;
              const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

              is_antiprompt = false;
              // Check if each of the reverse prompts appears at the end of the output.
              // If we're not running interactively, the reverse prompt might be tokenized with some following characters
              // so we'll compensate for that by widening the search window a bit.
              for (std::string & antiprompt : params.antiprompt) {
                  size_t extra_padding = params.interactive ? 0 : 2;
                  size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                      ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                      : 0;

                  if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                      if (params.interactive) {
                          is_interacting = true;
                      }
                      is_antiprompt = true;
                      break;
                  }
              }

              // check for reverse prompt using special tokens
              llama_token last_token = common_sampler_last(smpl);
              for (auto token : antiprompt_token) {
                  if (token == last_token) {
                      if (params.interactive) {
                          is_interacting = true;
                      }
                      is_antiprompt = true;
                      break;
                  }
              }

              if (is_antiprompt) {
                  LOG_DBG("found antiprompt: %s\n", last_output.c_str());
              }
          }

          // deal with end of generation tokens in interactive mode
          if (llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
              LOG_DBG("found an EOG token\n");

              if (params.interactive) {
                  if (!params.antiprompt.empty()) {
                      // tokenize and inject first reverse prompt
                      const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                      embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                      is_antiprompt = true;
                  }

                  if (params.enable_chat_template) {
                      chat_add_and_format("assistant", assistant_ss.str());
                  }
                  is_interacting = true;
                  LOG("\n");
              }
          }

          // if current token is not EOG, we add it to current assistant message
          if (params.conversation_mode) {
              const auto id = common_sampler_last(smpl);
              assistant_ss << common_token_to_piece(ctx, id, false);
          }

          if (n_past > 0 && is_interacting) {
              LOG_DBG("waiting for user input\n");

              if (params.conversation_mode) {
                  LOG("\n> ");
              }

              if (params.input_prefix_bos) {
                  LOG_DBG("adding input prefix BOS token\n");
                  embd_inp.push_back(llama_vocab_bos(vocab));
              }
              if (init_status == false) {
                std_msgs::msg::String::UniquePtr pub_string(
                  new std_msgs::msg::String());  
                pub_string->data = cute_words_;
                output_msg_publisher_->publish(std::move(pub_string));
                init_status = true;
              }
              {
                std::unique_lock<std::mutex> lg(mtx_llm_);
                task_permission_ = true;  // 占用
                lg.unlock();
              }
              std::string buffer = "";
              {
                std::unique_lock<std::mutex> lg(mtx_text_);
                cv_text_.wait(lg, [this]() { return user_prompt_ != "" || !rclcpp::ok(); });
                buffer = user_prompt_;
                user_prompt_ = "";
                lg.unlock();
              }              
              {
                std::unique_lock<std::mutex> lg(mtx_llm_);
                task_permission_ = false;
                sub_string = "";
                if (!result_string.empty()) {
                  ai_msgs::msg::PerceptionTargets::UniquePtr pub_data(
                    new ai_msgs::msg::PerceptionTargets());
                  ai_msgs::msg::Target target;
                  target.set__type(result_string);
                  pub_data->targets.emplace_back(std::move(target));
                  ai_msg_publisher_->publish(std::move(pub_data));
                }
                result_string = "";
                his_strings.clear();
                lg.unlock();
              }
              if (buffer.length() > 1) {
                  // append input suffix if any
                  if (!params.input_suffix.empty() && !params.conversation_mode) {
                      LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                      LOG("%s", params.input_suffix.c_str());
                  }

                  LOG_INF("buffer: '%s'\n", buffer.c_str());

                  const size_t original_size = embd_inp.size();

                  if (params.escape) {
                      string_process_escapes(buffer);
                  }

                  bool format_chat = params.conversation_mode && params.enable_chat_template;
                  std::string user_inp = format_chat
                      ? chat_add_and_format("user", std::move(buffer))
                      : std::move(buffer);
                  // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                  const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                  const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                  const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                  LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                  // if user stop generation mid-way, we must add EOT to finish model's last response
                  if (need_insert_eot && format_chat) {
                      llama_token eot = llama_vocab_eot(vocab);
                      embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                      need_insert_eot = false;
                  }

                  embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                  embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                  embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                  for (size_t i = original_size; i < embd_inp.size(); ++i) {
                      const llama_token token = embd_inp[i];
                      output_tokens.push_back(token);
                      output_ss << common_token_to_piece(ctx, token);
                  }

                  // reset assistant message
                  assistant_ss.str("");

                  n_remain -= line_inp.size();
                  LOG_DBG("n_remain: %d\n", n_remain);
              } else {
                  LOG_DBG("empty line, passing control back\n");
              }

              input_echo = false; // do not echo this again
          }

          if (n_past > 0) {
              if (is_interacting) {
                  common_sampler_reset(smpl);
              }
              is_interacting = false;
          }
      }

      // end of generation
      if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
          LOG(" [end of text]\n");
          break;
      }

      // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
      // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
      if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
          n_remain = params.n_predict;
          is_interacting = true;
      }
  }

  if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
      LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
      llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
  }

  LOG("\n\n");
  common_perf_print(ctx, smpl);

  common_sampler_free(smpl);

  llama_backend_free();

  ggml_threadpool_free_fn(threadpool);
  ggml_threadpool_free_fn(threadpool_batch);

  return 0;
}