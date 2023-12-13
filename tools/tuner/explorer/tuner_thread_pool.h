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
#ifndef TUNER_THREAD_POOL_H
#define TUNER_THREAD_POOL_H
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>
#include <condition_variable>

namespace tuner_ns {

class tuner_thread_pool {
public:
    tuner_thread_pool(size_t);
    ~tuner_thread_pool();

    template <typename F, typename... Args>
    auto add_task(F &&f, Args &&...args)
            -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    std::vector<std::thread> thread_pool;
    std::queue<std::function<void()>> task_queue;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop_pool;
};

template <typename F, typename... Args>
auto tuner_thread_pool::add_task(F &&f, Args &&...args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop_pool)
            throw std::runtime_error("add_task on stopped tuner_thread_pool");

        task_queue.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

} // namespace tuner_ns
#endif