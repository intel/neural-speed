//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "models/model_utils/pool.h"

serve_policy parse_serve_policy(const std::string& policy) {
  if (policy == "fcfs") {
    return serve_policy::FCFS;
  } else {
    fprintf(stderr, "Unexpected serve_policy %s!", policy.c_str());
    return serve_policy::UNKNOWN;
  }
}

// fcfs_pool
bool fcfs_pool::add(sequence seq) {
  context.emplace(seq);
  return true;
}

bool fcfs_pool::pop(sequence* seq) {
  if (empty()) {
    fprintf(stderr, "%s: pool is empty.\n", __func__);
    return false;
  }
  *seq = context.front();
  context.pop();
  return true;
}

void fcfs_pool::clear() {
  std::queue<sequence> empty_q;
  context.swap(empty_q);
}

bool fcfs_pool::empty() { return context.empty(); }

int fcfs_pool::size() { return context.size(); }

// serve_pool
serve_pool::serve_pool(const pool_property& property) {
  // default policy = FCFS
  std::lock_guard<std::mutex> lock(mtx);
  if (internel_pool != nullptr) return;
  internel_pool = new fcfs_pool(property);
}

serve_pool::serve_pool(const serve_policy& policy, const pool_property& property) {
  std::lock_guard<std::mutex> lock(mtx);
  if (internel_pool != nullptr) return;
  switch (policy) {
    case serve_policy::FCFS:
      internel_pool = new fcfs_pool(property);
    default:
      NE_ASSERT(false);
  }
}

serve_pool::~serve_pool() {
  std::lock_guard<std::mutex> lock(mtx);
  if (internel_pool != nullptr) {
    delete internel_pool;
  }
}

bool serve_pool::add(sequence seq) {
  std::lock_guard<std::mutex> lock(mtx);
  return internel_pool->add(std::move(seq));
}

bool serve_pool::pop(sequence* seq) {
  std::lock_guard<std::mutex> lock(mtx);
  return internel_pool->pop(seq);
}

void serve_pool::clear() {
  std::lock_guard<std::mutex> lock(mtx);
  internel_pool->clear();
}

bool serve_pool::empty() { return internel_pool->empty(); }

int serve_pool::size() { return internel_pool->size(); }
