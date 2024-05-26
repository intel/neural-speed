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
#pragma once
#include <map>
#include <thread>
#include <vector>
#include <sycl/sycl.hpp>

namespace bestla {

namespace sycl_device {

class SyclDevice {
 public:
  SyclDevice(bool profile) {
    // Create an exception handler for asynchronous SYCL exceptions
    static auto exception_handler = [](sycl::exception_list e_list) {
      for (std::exception_ptr const& e : e_list) {
        try {
          std::rethrow_exception(e);
        } catch (std::exception const& e) {
#if _DEBUG
          std::cout << "Failure" << std::endl;
#endif
          std::terminate();
        }
      }
    };

    auto d_selector{sycl::default_selector_v};
    if (profile) {
      sycl::property_list prop = {sycl::property::queue::enable_profiling()};
      mQueue = sycl::queue(d_selector, exception_handler, prop);
    } else {
      mQueue = sycl::queue(d_selector, exception_handler);
    }
  }

  inline sycl::queue* getQueue() { return &mQueue; }

  inline std::string getName() { return mQueue.get_device().get_info<sycl::info::device::name>(); };

  size_t getGlobalMemSize() { return mQueue.get_device().get_info<sycl::info::device::global_mem_size>(); }

  double getGlobalMemSizeGB() { return double(getGlobalMemSize()) / 1e9; }

  void print() {
    std::cout << "Running on device: " << mQueue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "EU count:" << mQueue.get_device().get_info<sycl::info::device::ext_intel_gpu_eu_count>()
              << "\n";  // 448
    std::cout << "EU count per subslice:"
              << mQueue.get_device().get_info<sycl::info::device::ext_intel_gpu_eu_count_per_subslice>() << "\n";  // 8
    std::cout << "EU SIMD width:" << mQueue.get_device().get_info<sycl::info::device::ext_intel_gpu_eu_simd_width>()
              << "\n";  // 8
    std::cout << "HW threads per EU:"
              << mQueue.get_device().get_info<sycl::info::device::ext_intel_gpu_hw_threads_per_eu>() << "\n";  // 8
    std::cout << "GPU slices:" << mQueue.get_device().get_info<sycl::info::device::ext_intel_gpu_slices>()
              << "\n";  // 7
    std::cout << "Subslice per slice:"
              << mQueue.get_device().get_info<sycl::info::device::ext_intel_gpu_subslices_per_slice>() << "\n";  // 8
    std::cout << "Global Memory size: " << getGlobalMemSizeGB() << "\n";                                           // 8
  }
  sycl::queue mQueue;
};

}  // namespace sycl_device
}  // namespace bestla
