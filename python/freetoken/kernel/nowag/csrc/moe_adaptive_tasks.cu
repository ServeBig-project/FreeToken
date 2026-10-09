#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kBaseBlockM = 16;
constexpr int kLargeBlockM = 64;

__global__ void build_adaptive_tasks_kernel(
    const int32_t* __restrict__ sorted_tickets,
    const int32_t* __restrict__ expert_ids16,
    const int32_t* __restrict__ num_tickets_post_padded,
    int2* __restrict__ tasks64,
    int2* __restrict__ tasks16,
    int32_t* __restrict__ task_counts,
    int num_routes,
    int num_experts,
    int sorted_capacity,
    int expert_block_capacity) {
  const int block16 = blockIdx.x * blockDim.x + threadIdx.x;
  const int active_blocks = min(
      min((*num_tickets_post_padded + kBaseBlockM - 1) / kBaseBlockM,
          sorted_capacity / kBaseBlockM),
      expert_block_capacity);
  if (block16 >= active_blocks) {
    return;
  }

  const int expert = expert_ids16[block16];
  if (expert < 0 || expert >= num_experts ||
      (block16 > 0 && expert_ids16[block16 - 1] == expert)) {
    return;
  }

  int run_end = block16 + 1;
  while (run_end < active_blocks && expert_ids16[run_end] == expert) {
    ++run_end;
  }
  const int row_start = block16 * kBaseBlockM;
  const int row_end = run_end * kBaseBlockM;
  int real_routes = 0;
  for (int row = row_start; row < row_end; ++row) {
    const int ticket = sorted_tickets[row];
    real_routes += ticket >= 0 && ticket < num_routes;
  }
  if (real_routes == 0) {
    return;
  }

  const int large_tasks = real_routes / kLargeBlockM;
  const int remainder = real_routes - large_tasks * kLargeBlockM;
  const int small_tasks =
      (remainder + kBaseBlockM - 1) / kBaseBlockM;
  if (large_tasks > 0) {
    const int output = atomicAdd(task_counts, large_tasks);
    for (int index = 0; index < large_tasks; ++index) {
      tasks64[output + index] =
          make_int2(row_start + index * kLargeBlockM, expert);
    }
  }
  if (small_tasks > 0) {
    const int output = atomicAdd(task_counts + 1, small_tasks);
    const int remainder_start = row_start + large_tasks * kLargeBlockM;
    for (int index = 0; index < small_tasks; ++index) {
      tasks16[output + index] =
          make_int2(remainder_start + index * kBaseBlockM, expert);
    }
  }
}

__global__ void build_adaptive_tasks_tail64_kernel(
    const int32_t* __restrict__ sorted_tickets,
    const int32_t* __restrict__ expert_ids16,
    const int32_t* __restrict__ num_tickets_post_padded,
    int2* __restrict__ tasks64,
    int2* __restrict__ tasks16,
    int32_t* __restrict__ task_counts,
    int num_routes,
    int num_experts,
    int sorted_capacity,
    int expert_block_capacity) {
  const int block16 = blockIdx.x * blockDim.x + threadIdx.x;
  const int active_blocks = min(
      min((*num_tickets_post_padded + kBaseBlockM - 1) / kBaseBlockM,
          sorted_capacity / kBaseBlockM),
      expert_block_capacity);
  if (block16 >= active_blocks) {
    return;
  }

  const int expert = expert_ids16[block16];
  if (expert < 0 || expert >= num_experts ||
      (block16 > 0 && expert_ids16[block16 - 1] == expert)) {
    return;
  }

  int run_end = block16 + 1;
  while (run_end < active_blocks && expert_ids16[run_end] == expert) {
    ++run_end;
  }
  const int row_start = block16 * kBaseBlockM;
  const int row_end = run_end * kBaseBlockM;
  int real_routes = 0;
  for (int row = row_start; row < row_end; ++row) {
    const int ticket = sorted_tickets[row];
    real_routes += ticket >= 0 && ticket < num_routes;
  }
  if (real_routes == 0) {
    return;
  }

  const int full64 = real_routes / kLargeBlockM;
  const int remainder = real_routes - full64 * kLargeBlockM;
  const bool tail64 = remainder >= 49;
  const int count64 = full64 + static_cast<int>(tail64);
  if (count64 > 0) {
    const int output = atomicAdd(task_counts, count64);
    for (int index = 0; index < full64; ++index) {
      tasks64[output + index] =
          make_int2(row_start + index * kLargeBlockM, expert);
    }
    if (tail64) {
      tasks64[output + full64] =
          make_int2(row_start + full64 * kLargeBlockM, expert);
    }
  }

  const int small_tasks =
      tail64 ? 0 : (remainder + kBaseBlockM - 1) / kBaseBlockM;
  if (small_tasks > 0) {
    const int output = atomicAdd(task_counts + 1, small_tasks);
    const int remainder_start = row_start + full64 * kLargeBlockM;
    for (int index = 0; index < small_tasks; ++index) {
      tasks16[output + index] =
          make_int2(remainder_start + index * kBaseBlockM, expert);
    }
  }
}

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

}  // namespace

void nowag_moe_build_adaptive_tasks_cuda(
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor tasks64,
    torch::Tensor tasks16,
    torch::Tensor task_counts,
    int64_t num_routes,
    int64_t num_experts) {
  TORCH_CHECK(num_routes > 0, "num_routes must be positive");
  TORCH_CHECK(num_experts > 0, "num_experts must be positive");
  check_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_tensor(expert_ids16, "expert_ids16", at::kInt);
  check_tensor(
      num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  check_tensor(tasks64, "tasks64", at::kInt);
  check_tensor(tasks16, "tasks16", at::kInt);
  check_tensor(task_counts, "task_counts", at::kInt);
  TORCH_CHECK(sorted_tickets.dim() == 1 && sorted_tickets.numel() >= 16,
              "sorted_tickets must be a non-empty BM16-aligned buffer");
  TORCH_CHECK(expert_ids16.dim() == 1 && expert_ids16.numel() > 0,
              "expert_ids16 must be a non-empty block-owner buffer");
  TORCH_CHECK(num_tickets_post_padded.numel() == 1,
              "num_tickets_post_padded must contain one device scalar");
  TORCH_CHECK(tasks64.dim() == 2 && tasks64.size(1) == 2,
              "tasks64 must be [capacity64,2]");
  TORCH_CHECK(tasks16.dim() == 2 && tasks16.size(1) == 2,
              "tasks16 must be a residual-task queue [capacity16,2]");
  TORCH_CHECK(task_counts.dim() == 1 && task_counts.numel() == 2,
              "task_counts must be int32[2]");
  const int64_t required64 = std::max<int64_t>(1, num_routes / 64);
  const int64_t active_experts = std::min(num_routes, num_experts);
  const int64_t residual_bound =
      (num_routes + 15 * active_experts) / 16;
  const int64_t required16 = std::max<int64_t>(
      1,
      std::min(
          std::min(num_routes, 4 * num_experts), residual_bound));
  TORCH_CHECK(tasks64.size(0) >= required64,
              "tasks64 does not have the static route capacity");
  TORCH_CHECK(tasks16.size(0) >= required16,
              "tasks16 must provide at least ", required16,
              " residual-task rows");

  const auto device = sorted_tickets.device();
  const torch::Tensor tensors[] = {
      expert_ids16, num_tickets_post_padded, tasks64, tasks16, task_counts};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device,
                "all adaptive task tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      task_counts.data_ptr<int32_t>(), 0, 2 * sizeof(int32_t), stream));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(
      (expert_ids16.numel() + kThreads - 1) / kThreads);
  build_adaptive_tasks_kernel<<<blocks, kThreads, 0, stream>>>(
      sorted_tickets.data_ptr<int32_t>(),
      expert_ids16.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      reinterpret_cast<int2*>(tasks64.data_ptr<int32_t>()),
      reinterpret_cast<int2*>(tasks16.data_ptr<int32_t>()),
      task_counts.data_ptr<int32_t>(),
      static_cast<int>(num_routes),
      static_cast<int>(num_experts),
      static_cast<int>(sorted_tickets.numel()),
      static_cast<int>(expert_ids16.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nowag_moe_build_adaptive_tasks_tail64_cuda(
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor tasks64,
    torch::Tensor tasks16,
    torch::Tensor task_counts,
    int64_t num_routes,
    int64_t num_experts) {
  TORCH_CHECK(num_routes > 0, "num_routes must be positive");
  TORCH_CHECK(num_experts > 0, "num_experts must be positive");
  check_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_tensor(expert_ids16, "expert_ids16", at::kInt);
  check_tensor(
      num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  check_tensor(tasks64, "tasks64", at::kInt);
  check_tensor(tasks16, "tasks16", at::kInt);
  check_tensor(task_counts, "task_counts", at::kInt);
  TORCH_CHECK(sorted_tickets.dim() == 1 && sorted_tickets.numel() >= 16,
              "sorted_tickets must be a non-empty BM16-aligned buffer");
  TORCH_CHECK(expert_ids16.dim() == 1 && expert_ids16.numel() > 0,
              "expert_ids16 must be a non-empty block-owner buffer");
  TORCH_CHECK(num_tickets_post_padded.numel() == 1,
              "num_tickets_post_padded must contain one device scalar");
  TORCH_CHECK(tasks64.dim() == 2 && tasks64.size(1) == 2,
              "tasks64 must be [capacity64,2]");
  TORCH_CHECK(tasks16.dim() == 2 && tasks16.size(1) == 2,
              "tasks16 must be a residual-task queue [capacity16,2]");
  TORCH_CHECK(task_counts.dim() == 1 && task_counts.numel() == 2,
              "task_counts must be int32[2]");
  const int64_t required64 = std::max<int64_t>(1, num_routes / 49);
  const int64_t active_experts = std::min(num_routes, num_experts);
  const int64_t residual_bound =
      (num_routes + 15 * active_experts) / 16;
  const int64_t required16 = std::max<int64_t>(
      1,
      std::min(
          std::min(num_routes, 4 * num_experts), residual_bound));
  TORCH_CHECK(tasks64.size(0) >= required64,
              "tasks64 does not have the tail64 static route capacity");
  TORCH_CHECK(tasks16.size(0) >= required16,
              "tasks16 must provide at least ", required16,
              " residual-task rows");

  const auto device = sorted_tickets.device();
  const torch::Tensor tensors[] = {
      expert_ids16, num_tickets_post_padded, tasks64, tasks16, task_counts};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device,
                "all tail64 task tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      task_counts.data_ptr<int32_t>(), 0, 2 * sizeof(int32_t), stream));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>(
      (expert_ids16.numel() + kThreads - 1) / kThreads);
  build_adaptive_tasks_tail64_kernel<<<blocks, kThreads, 0, stream>>>(
      sorted_tickets.data_ptr<int32_t>(),
      expert_ids16.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      reinterpret_cast<int2*>(tasks64.data_ptr<int32_t>()),
      reinterpret_cast<int2*>(tasks16.data_ptr<int32_t>()),
      task_counts.data_ptr<int32_t>(),
      static_cast<int>(num_routes),
      static_cast<int>(num_experts),
      static_cast<int>(sorted_tickets.numel()),
      static_cast<int>(expert_ids16.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
