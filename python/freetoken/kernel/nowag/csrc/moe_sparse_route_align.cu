#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kMaxRoutes = 256;
constexpr uint64_t kInvalidKey = ~uint64_t{0};

struct SparseAlignShared {
  uint64_t keys[kMaxRoutes];
  int32_t scatter_positions[kMaxRoutes];
  int32_t block_experts[kMaxRoutes];
  int32_t padded_routes;
  int32_t active_blocks;
};

__global__ void sparse_route_align_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_tickets,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ num_tickets_post_padded,
    int num_routes,
    int block_size,
    int num_experts) {
  __shared__ SparseAlignShared shared;
  const int thread = threadIdx.x;

  uint64_t key = kInvalidKey;
  if (thread < num_routes) {
    const int expert = topk_ids[thread];
    if (expert >= 0 && expert < num_experts) {
      key = (static_cast<uint64_t>(static_cast<uint32_t>(expert)) << 32) |
          static_cast<uint32_t>(thread);
    }
  }
  shared.keys[thread] = key;

  const int sorted_capacity = num_routes * block_size;
  for (int index = thread; index < sorted_capacity; index += blockDim.x) {
    sorted_tickets[index] = num_routes;
  }
  for (int index = thread; index < num_routes; index += blockDim.x) {
    expert_ids[index] = num_experts;
  }
  __syncthreads();

  // A route's flattened ticket is the low half of the key, so this network
  // orders routes by (expert, ticket).  Invalid routes sort to the end.
  for (int sequence = 2; sequence <= blockDim.x; sequence <<= 1) {
    for (int stride = sequence >> 1; stride > 0; stride >>= 1) {
      const int partner = thread ^ stride;
      if (partner > thread) {
        const uint64_t left = shared.keys[thread];
        const uint64_t right = shared.keys[partner];
        const bool ascending = (thread & sequence) == 0;
        if ((ascending && left > right) || (!ascending && left < right)) {
          shared.keys[thread] = right;
          shared.keys[partner] = left;
        }
      }
      __syncthreads();
    }
  }

  if (thread == 0) {
    int route = 0;
    int padded_cursor = 0;
    int block_cursor = 0;
    while (route < num_routes && shared.keys[route] != kInvalidKey) {
      const int expert = static_cast<int>(shared.keys[route] >> 32);
      int run_end = route + 1;
      while (run_end < num_routes &&
             shared.keys[run_end] != kInvalidKey &&
             static_cast<int>(shared.keys[run_end] >> 32) == expert) {
        ++run_end;
      }

      const int run_size = run_end - route;
      const int padded_run_size =
          ((run_size + block_size - 1) / block_size) * block_size;
      for (int index = route; index < run_end; ++index) {
        shared.scatter_positions[index] = padded_cursor + index - route;
      }
      const int run_blocks = padded_run_size / block_size;
      for (int index = 0; index < run_blocks; ++index) {
        shared.block_experts[block_cursor + index] = expert;
      }
      padded_cursor += padded_run_size;
      block_cursor += run_blocks;
      route = run_end;
    }
    shared.padded_routes = padded_cursor;
    shared.active_blocks = block_cursor;
  }
  __syncthreads();

  if (thread < num_routes && shared.keys[thread] != kInvalidKey) {
    sorted_tickets[shared.scatter_positions[thread]] =
        static_cast<int32_t>(shared.keys[thread]);
  }
  if (thread < shared.active_blocks) {
    expert_ids[thread] = shared.block_experts[thread];
  }
  if (thread == 0) {
    num_tickets_post_padded[0] = shared.padded_routes;
  }
}

int route_sort_threads(int64_t num_routes) {
  int threads = 32;
  while (threads < num_routes) {
    threads <<= 1;
  }
  return threads;
}

}  // namespace

std::vector<torch::Tensor> nowag_moe_sparse_route_align_cuda(
    torch::Tensor topk_ids,
    int64_t block_size,
    int64_t num_experts) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be a CUDA tensor");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(topk_ids.scalar_type() == at::kInt,
              "topk_ids must have dtype int32");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be [M,T]");
  const int64_t num_routes = topk_ids.numel();
  TORCH_CHECK(num_routes <= kMaxRoutes,
              "topk_ids must contain at most 256 routes");
  TORCH_CHECK(block_size == 16 || block_size == 32 || block_size == 64 ||
                  block_size == 128,
              "block_size must be 16, 32, 64, or 128");
  TORCH_CHECK(num_experts > 0 &&
                  num_experts <= std::numeric_limits<int32_t>::max(),
              "num_experts must fit a positive int32");

  c10::cuda::CUDAGuard device_guard(topk_ids.device());
  auto sorted_tickets = torch::empty(
      {num_routes * block_size}, topk_ids.options());
  auto expert_ids = torch::empty({num_routes}, topk_ids.options());
  auto num_tickets_post_padded = torch::empty({1}, topk_ids.options());

  const int threads = route_sort_threads(num_routes);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  sparse_route_align_kernel<<<1, threads, 0, stream>>>(
      topk_ids.data_ptr<int32_t>(),
      sorted_tickets.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      static_cast<int>(num_routes),
      static_cast<int>(block_size),
      static_cast<int>(num_experts));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {sorted_tickets, expert_ids, num_tickets_post_padded};
}
