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

#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string& input, char delimiter);

std::string getRandomSubword(const std::string& input);

std::tuple<std::string, int, int, int, int> parse_detection_string(const std::string& input);

#endif  // UTIL_H
