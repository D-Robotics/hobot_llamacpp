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

#include "include/util.h"

// 分割函数
std::vector<std::string> split(const std::string& input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
      if (!token.empty()) {
          result.push_back(token);
      }
  }
  return result;
}

// 随机选择子词
std::string getRandomSubword(const std::string& input) {
  std::vector<std::string> words = split(input, ';');
  if (words.empty()) return "";

  // 使用随机数生成器
  static std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_int_distribution<size_t> dist(0, words.size() - 1);

  return words[dist(rng)];
}

std::tuple<std::string, int, int, int, int> parse_detection_string(const std::string& input) {
  std::regex ref_regex("<ref>(.*?)</ref>");
  std::regex box_regex("<box>\\[\\[(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)\\]\\]</box>");

  std::smatch ref_match;
  std::smatch box_match;

  std::string category = "";
  int x1 = -1, y1 = -1, x2 = -1, y2 = -1;

  if (std::regex_search(input, ref_match, ref_regex)) {
      category = ref_match[1].str();
  }

  if (std::regex_search(input, box_match, box_regex)) {
      x1 = std::stoi(box_match[1]);
      y1 = std::stoi(box_match[2]);
      x2 = std::stoi(box_match[3]);
      y2 = std::stoi(box_match[4]);
  }

  return {category, x1, y1, x2, y2};
}