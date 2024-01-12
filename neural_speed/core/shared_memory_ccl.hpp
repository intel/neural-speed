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
#include <algorithm>
#include <assert.h>
#include <immintrin.h>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "oneapi/ccl.hpp"
#include "layers/ele_wise.h"

// states for collectives
enum ccl_state {
  ccl_begin = 0,
  copy_in_done,
  reduce_done,
  copy_out_done,
};
#ifndef _WIN32
void* shared_open(const char* name, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    return mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
  } else {
    printf("shared_open %s failed\n", name);
    return nullptr;
  }
}

void* shared_create(const char* name, void* bytes, size_t nbytes) {
  int d = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if ((d != -1) && (nbytes = write(d, bytes, nbytes))) {
    return mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, d, 0);
  } else {
    printf("shared_create %s failed\n", name);
    return nullptr;
  }
}

void shared_close(const char* name, void* bytes, size_t nbytes) {
  int d = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
  if (d != -1) {
    munmap(bytes, nbytes);
    shm_unlink(name);
  }
}
#endif

static constexpr size_t CCL_BUF_SIZE = 1048576;
struct ccl_buffer {
  enum ccl_state state;
  char data[CCL_BUF_SIZE];
};
struct ccl_buffer* cbuffer;

void wait_state_equal(int index, enum ccl_state state) {
  volatile enum ccl_state* state_ptr = &(cbuffer[index].state);
  while (*state_ptr != state) {
    _mm_pause();
  }
}

void wait_state_change(int index, enum ccl_state state) {
  volatile enum ccl_state* state_ptr = &(cbuffer[index].state);
  while (*state_ptr == state) {
    _mm_pause();
  }
}

void reduce_fp32_buffers(int num_elements, int num_buffers, struct ccl_buffer* cbuffer) {
  auto rank_0 = reinterpret_cast<float*>(cbuffer[0].data);
  // all buffers reduce to rank 0 and then broadcast
  for (int i = 1; i < num_buffers; ++i) {
    ne_vec_add_f32(num_elements, rank_0, rank_0, reinterpret_cast<float*>(cbuffer[i].data));
  }
}

void reduce_buffers(struct ccl_buffer* cbuffer, int num_elements, int num_buffers) {
  // TODO(chenxi) only support fp32 reduce, add other data type if needed
  if (num_buffers >= 2) {
    reduce_fp32_buffers(num_elements, num_buffers, cbuffer);
  } else {
    assert(!"Not supported buffer number.");
  }
}

void shm_all_reduce(float* sendBuf, float* recvBuf, size_t count, size_t rank, size_t world_size) {
  for (int offset = 0; offset < count * sizeof(float); offset += CCL_BUF_SIZE) {
    auto send_ptr = reinterpret_cast<char*>(sendBuf) + offset;
    auto recv_ptr = reinterpret_cast<char*>(recvBuf) + offset;
    size_t chunk_size = std::min(count * sizeof(float) - offset, (size_t)CCL_BUF_SIZE);
    size_t chunk_count = chunk_size / sizeof(float);

    memcpy(cbuffer[rank].data, send_ptr, chunk_size);
    cbuffer[rank].state = copy_in_done;

    if (rank == 0) {
      // compute allreduce result on rank 0
      for (int i = 1; i < world_size; i++) {
        // wait until the other rank copy the buffer
        wait_state_equal(i, copy_in_done);
      }
      reduce_buffers(cbuffer, chunk_count, world_size);
      cbuffer[rank].state = reduce_done;
      memcpy(recv_ptr, cbuffer[0].data, chunk_size);
    }
    if (rank != 0) {
      wait_state_equal(0, reduce_done);
      memcpy(recv_ptr, cbuffer[0].data, chunk_size);
      cbuffer[rank].state = copy_out_done;
    }
    if (rank == 0) {
      for (int i = 1; i < world_size; i++) {
        wait_state_equal(i, copy_out_done);
      }
      cbuffer[rank].state = ccl_begin;
    }
    if (rank != 0) {
      // if rank 0 spin too fast it could be in state 1 of next allreduce
      // in this case wait_state_change(0, 0) may cause deadlock
      // what we are certain is when rank 0 finishes the state won't be 2
      wait_state_change(0, reduce_done);
      cbuffer[rank].state = ccl_begin;
    }
  }
}
