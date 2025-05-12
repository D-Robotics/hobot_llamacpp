[English](./README.md) | 简体中文

Getting Started with hobot llamacpp
=======


# 功能介绍

hobot llamacpp package是基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 集成集语言大模型 LLM, 多模态大模型 VLM 为一体的使用示例。主要功能有两种：

- 纯语言模型 LLM：支持系统提示词设定, 文本输入, 文本输出对话。其中文本可通过参数配置, 或运行中通过string msg 话题消息实时控制。最终输出文本, 通过 string msg 话题消息发出。

- 视觉语言模型 VLM：支持系统提示词设定, 文本和图片输入, 文本输出对话。其中文本可通过参数配置, 或运行中通过string msg 话题消息实时控制。图像数据来源于本地图片回灌和订阅到的image msg。最终输出文本, 通过 string msg 话题消息发出。

# 开发环境

- 编程语言: C/C++
- 开发平台: X5
- 系统版本：Ubuntu 22.04
- 编译工具链: Linux GCC 11.4.0

# 编译

- X5版本：支持在X5 Ubuntu系统上编译。

- S100版本：支持在S100 Ubuntu系统上编译。

同时支持通过编译选项控制编译pkg的依赖和pkg的功能。

## 依赖库

- opencv:3.4.5

ros package：

- dnn node
- cv_bridge
- sensor_msgs
- hbm_img_msgs
- ai_msgs

hbm_img_msgs为自定义的图片消息格式, 用于shared mem场景下的图片传输, hbm_img_msgs pkg定义在hobot_msgs中, 因此如果使用shared mem进行图片传输, 需要依赖此pkg。


## 编译选项

1、SHARED_MEM

- shared mem（共享内存传输）使能开关, 默认打开（ON）, 编译时使用-DSHARED_MEM=OFF命令关闭。
- 如果打开, 编译和运行会依赖hbm_img_msgs pkg, 并且需要使用tros进行编译。
- 如果关闭, 编译和运行不依赖hbm_img_msgs pkg, 支持使用原生ros和tros进行编译。
- 对于shared mem通信方式, 当前只支持订阅nv12格式图片。

## 板端 Ubuntu系统上编译

1、编译环境确认

- 板端已安装X5/S100 Ubuntu系统。
- 当前编译终端已设置TogetherROS环境变量：`source PATH/setup.bash`。其中PATH为TogetherROS的安装路径。
- 已安装ROS2编译工具colcon。安装的ROS不包含编译工具colcon, 需要手动安装colcon。colcon安装命令：`pip install -U colcon-common-extensions`
- 已编译dnn node package

2、编译依赖

- 链接第三方仓库 [llama.cpp](https://github.com/ggml-org/llama.cpp):
 
```shell
cmake -B build
cmake --build build --config Release
# 链接llama.cpp到工程目录下
cd hobot_llamacpp && ln -s thirdparty/llama.cpp llama.cpp
```

3、编译

- 编译命令：

```shell
# RDK X5
colcon build --merge-install --cmake-args -DPLATFORM_X5=ON --packages-select hobot_llamacpp
```

```shell
# RDK S100
colcon build --merge-install --cmake-args -DPLATFORM_S100=ON --packages-select hobot_llamacpp
```

# 使用介绍

## 依赖

- mipi_cam package：发布图片msg
- usb_cam package：发布图片msg
- websocket package：显示图片msg

## 参数

| 参数名             | 解释                                  | 是否必须             | 默认值              | 备注                                                                    |
| ------------------ | ------------------------------------- | -------------------- | ------------------- | ----------------------------------------------------------------------- |
| feed_type          | 数据来源, 0：VLM 本地；1：VLM 订阅; 2: LLM 订阅           | 否                   | 0                   |                                                                         |
| image              | 本地图片地址                          | 否                   | config/image2.jpg     |                                                                         |
| is_shared_mem_sub  | 使用shared mem通信方式订阅图片        | 否                   | 0                   |                                                                         |
| llm_threads | 语言模型推理线程数 | 否 | 8 | |
| model_file_name | 视觉模型模型名 | 否 | "vit_model_int16_v2.bin"" | |
| llm_model_name | 语言模型模型名 | 否 | "Qwen2.5-0.5B-Instruct-Q4_0.gguf" | |
| user_prompt | 语言模型文本提示词 | 否 | "" | |
| system_prompt | 语言模型系统提示词 | 否 | "You are a helpful assistant." | |
| pre_infer | 提前推理开关 | 否 | 0 | |
| ai_msg_pub_topic_name | 发布智能结果的topicname | 否                   | /llama_cpp_node | |
| text_msg_pub_topic_name | 发布智能结果的topicname,中间结果 | 否                   | /tts_text | |
| ros_img_sub_topic_name | 接收ros图片话题名 | 否                   | /image | |
| ros_string_sub_topic_name | 接收string消息话题获得文本提示词 | 否                   | /prompt_text | |


## 使用说明

- 发布提示词：hobot_llamacpp 依赖string msg话题消息获取提示词。string msg话题使用示例如下。其中 /prompt_text 为话题名。data字段中的数据为string字符串, 设置语言模型提示词。

```
ros2 topic pub /prompt_text std_msgs/msg/String "{data: '请描述这张图片'}"
```

## 运行

- hobot_llamacpp 使用到的模型需要在模型仓库中下载。

  - [图像编码模型](https://huggingface.co/D-Robotics/InternVL2_5-1B-GGUF-BPU/blob/main/rdkx5/vit_model_int16_v2.bin)

  - [语言编解码模型](https://huggingface.co/D-Robotics/InternVL2_5-1B-GGUF-BPU/blob/main/Qwen2.5-0.5B-Instruct-Q4_0.gguf)

- 编译成功后, 将生成的install路径拷贝到地平线RDK上（如果是在RDK上编译, 忽略拷贝步骤）, 并执行如下命令运行。

## X5 Ubuntu系统上运行

运行方式1, 使用可执行文件启动：
```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/local_setup.bash
# config中为示例使用的模型, 回灌使用的本地图片
# 根据实际安装路径进行拷贝（docker中的安装路径为install/lib/hobot_llamacpp/config/, 拷贝命令为cp -r install/lib/hobot_llamacpp/config/ .）。
cp -r install/lib/hobot_llamacpp/config/ .

# 运行模式1：
# 使用本地jpg格式图片进行回灌预测, 输入自定义用户提示词
ros2 run hobot_llamacpp hobot_llamacpp --ros-args -p feed_type:=0 -p image:=config/image2.jpg -p image_type:=0 -p user_prompt:="描述一下这张图片."

# 运行模式2：
# 使用订阅到的image msg(topic为/image)进行预测, 设置受控话题名(/prompt_text)为并设置log级别为warn。同时在另一个窗口发送string话题(topic为/prompt_text) 变更用户提示词
ros2 run hobot_llamacpp hobot_llamacpp --ros-args -p feed_type:=1 --ros-args --log-level warn -p ros_string_sub_topic_name:="/prompt_text"

ros2 topic pub /prompt_text std_msgs/msg/String "{data: '描述一下这张图片.'}"

# 运行模式3：使用语言模型进行推理交互
ros2 run hobot_llamacpp hobot_llamacpp --ros-args -p feed_type:=2 -p system_prompt:="config/system_prompt.txt" --ros-args --log-level warn

ros2 topic pub /prompt_text std_msgs/msg/String "{data: '周末应该怎么休息?'}"
```

运行方式2, 使用launch文件启动：
```shell
export COLCON_CURRENT_PREFIX=./install
source ./install/setup.bash
# config中为示例使用的模型, 根据实际安装路径进行拷贝
# 如果是板端编译（无--merge-install编译选项）, 拷贝命令为cp -r install/PKG_NAME/lib/PKG_NAME/config/ ., 其中PKG_NAME为具体的package名。

# 配置MIPI摄像头
export CAM_TYPE=mipi

# 启动launch文件, 使用F37 sensor通过shared mem方式发布nv12格式图片
ros2 launch hobot_llamacpp llama_vlm.launch.py
```

## X5 yocto系统上运行

```shell
export ROS_LOG_DIR=/userdata/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./install/lib/

# config中为示例使用的模型, 回灌使用的本地图片
cp -r install/lib/hobot_llamacpp/config/ .

# 运行模式1：
# 使用本地jpg格式图片进行回灌预测, 输入自定义用户提示词
./install/lib/hobot_llamacpp/hobot_llamacpp --ros-args -p feed_type:=0 -p image:=config/image2.jpg -p image_type:=0 -p user_prompt:="描述一下这张图片."

# 运行模式2：
# 使用订阅到的image msg(topic为/image)进行预测, 设置受控话题名(/prompt_text)为并设置log级别为warn。同时在另一个窗口发送string话题(topic为/prompt_text) 变更检测类别
./install/lib/hobot_llamacpp/hobot_llamacpp --ros-args -p feed_type:=1 --ros-args --log-level warn -p ros_string_sub_topic_name:="/prompt_text"

ros2 topic pub /prompt_text std_msgs/msg/String "{data: '描述一下这张图片.'}"

# 运行模式3：使用语言模型进行推理
./install/lib/hobot_llamacpp/hobot_llamacpp --ros-args -p feed_type:=2 -p system_prompt:="config/system_prompt.txt" --ros-args --log-level warn

ros2 topic pub /prompt_text std_msgs/msg/String "{data: '周末应该怎么休息?'}"
```

# 结果分析

## X5结果展示

### 视觉语言模型

运行命令：`ros2 run hobot_llamacpp hobot_llamacpp --ros-args -p feed_type:=0 -p image:=config/image2.jpg -p image_type:=0 -p user_prompt:="描述一下这张图片."`

```bash
[WARN] [1744635572.183177153] [llama_cpp_node]: Create ai msg publisher with topic_name: /llama_cpp_node
[INFO] [1744635572.214685272] [llama_cpp_node]: Dnn node feed with local image: image2.jpg
[INFO] [1744635574.739131628] [llama_cpp_node]: Output from frame_id: feedback, stamp: 0.0
llama_init_from_model: n_seq_max     = 1
llama_init_from_model: n_ctx         = 4096
llama_init_from_model: n_ctx_per_seq = 4096
llama_init_from_model: n_batch       = 2048
llama_init_from_model: n_ubatch      = 512
llama_init_from_model: flash_attn    = 0
llama_init_from_model: freq_base     = 1000000.0
llama_init_from_model: freq_scale    = 1
llama_init_from_model: n_ctx_per_seq (4096) < n_ctx_train (32768) -- the full capacity of the model will not be utilized
llama_kv_cache_init: kv_size = 4096, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 24, can_shift = 1
llama_kv_cache_init:        CPU KV buffer size =    48.00 MiB
llama_init_from_model: KV self size  =   48.00 MiB, K (f16):   24.00 MiB, V (f16):   24.00 MiB
llama_init_from_model:        CPU  output buffer size =     0.58 MiB
llama_init_from_model:        CPU compute buffer size =   299.74 MiB
llama_init_from_model: graph nodes  = 846
llama_init_from_model: graph splits = 1

这张图片展示了一只熊猫的场景。熊猫正趴在地上，周围有竹子。熊猫的耳朵竖起，眼睛睁大，看起来非常专注。背景中可以看到一些绿色的植物，可能是竹子。整个场景给人一种在自然环境中拍摄的感觉，可能是在动物园或野生动物保护区。熊猫的毛色是典型的黑白相间，非常可爱。
llama_perf_context_print:        load time =   11133.16 ms
llama_perf_context_print: prompt eval time =    2230.69 ms /   282 tokens (    7.91 ms per token,   126.42 tokens per second)
llama_perf_context_print:        eval time =    3745.39 ms /    74 runs   (   50.61 ms per token,    19.76 tokens per second)
llama_perf_context_print:       total time =   15033.11 ms /   356 tokens
```

日志结果:
![image](img/vlm_result.png)

### 语言模型

运行命令：`ros2 run hobot_llamacpp hobot_llamacpp --ros-args -p feed_type:=2 -p system_prompt:="config/system_prompt.txt" -p user_prompt:="周末应该怎么休息?"`

```bash
system_info: n_threads = 8 (n_threads_batch = 8) / 8 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

prompt: "<|im_start|>system
你是一名人工智能助手。<|im_end|>
"
tokens: [ '<|im_start|>':151644, 'system':8948, '':198, '':56568, '':110124, '':104455, '':110498, '':1773, '<|im_end|>':151645, '':198 ]
Chat: interactive mode on.
sampler seed: 1302666639
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 4096
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.500
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist
generate: n_ctx = 4096, n_batch = 2048, n_predict = 128, n_keep = 0

> '周末应该怎么休息?'
 休息很重要，可以看看书、听音乐、画画、运动
```
