/*******************************************************************************
 * Copyright 2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#include "tuner_thread_pool.h"

namespace tuner_ns {

// the constructor just launches some amount of thread_pool
tuner_thread_pool::tuner_thread_pool(size_t thread_num) : stop_pool(false) {
    for (size_t i = 0; i < thread_num; ++i)
        thread_pool.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop_pool || !this->task_queue.empty();
                    });
                    if (this->stop_pool && this->task_queue.empty()) return;
                    task = std::move(this->task_queue.front());
                    this->task_queue.pop();
                }

                task();
            }
        });
}

tuner_thread_pool::~tuner_thread_pool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop_pool = true;
    }
    condition.notify_all();
    for (std::thread &cur_thread : thread_pool)
        cur_thread.join();
}
} // namespace tuner_ns