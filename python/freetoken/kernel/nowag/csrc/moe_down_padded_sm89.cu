#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "nowag_k48_core.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpOutputN = 16;
constexpr int kGroupSize = nowag::k48_core::kGroupSize;
constexpr int kPaddedGroupSize = nowag::k48_core::kPaddedGroupSize;
constexpr int kGroupsPerPacket = nowag::k48_core::kGroupsPerPacket;
constexpr int kExactPacketK = nowag::k48_core::kExactPacketK;
constexpr int kPhysicalPacketK = nowag::k48_core::kPhysicalPacketK;
constexpr int kGroupScheduleM = 8;

using nowag::k48_core::compute_coalesced_k48_packet;
using nowag::k48_core::compute_exact_k48_packet;
using nowag::k48_core::compute_padded_packet;
using nowag::k48_core::swizzled_a_offset;

__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
  return __float2bfloat16_rn(value);
}

__global__ void apply_down_input_norm_kernel(
    __nv_bfloat16* sorted_middle,
    const __nv_bfloat16* input_norm,
    const int32_t* expert_ids,
    const int32_t* num_tickets_post_padded,
    int rows,
    int physical_width,
    int logical_width,
    int input_group_start_lane,
    int block_m) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t total = static_cast<int64_t>(rows) * physical_width;
  if (linear >= total) {
    return;
  }
  const int row = linear / physical_width;
  if (row >= *num_tickets_post_padded) {
    return;
  }
  const int physical_k = linear - static_cast<int64_t>(row) * physical_width;
  const int logical_k = physical_k - input_group_start_lane;
  if (logical_k < 0 || logical_k >= logical_width) {
    sorted_middle[linear] = from_float(0.0f);
    return;
  }
  const int expert = expert_ids[row / block_m];
  const float value = __bfloat162float(sorted_middle[linear]);
  const float scale = __bfloat162float(
      input_norm[static_cast<int64_t>(expert) * logical_width + logical_k]);
  sorted_middle[linear] = from_float(value * scale);
}

__device__ __forceinline__ void cp_async_ca_4(
    void* shared_destination,
    const void* global_source,
    int source_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::
          "r"(shared_address),
          "l"(global_source),
          "r"(source_bytes));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

template <int BlockM, int NumWarps>
__device__ __forceinline__ void initialize_padded_a_buffers(
    __nv_bfloat16* shared_a) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane < kGroupsPerPacket) {
    const int group_in_packet = lane;
    const int shared_k = group_in_packet * kPaddedGroupSize + 6;
    for (int local_m = warp; local_m < BlockM; local_m += NumWarps) {
#pragma unroll
      for (int pipe = 0; pipe < 2; ++pipe) {
        auto* destination = reinterpret_cast<uint32_t*>(
            shared_a + pipe * BlockM * kPhysicalPacketK +
            swizzled_a_offset(local_m, shared_k));
        *destination = 0;
      }
    }
  }
}

template <int BlockM, int NumWarps, bool FullPacket>
__device__ __forceinline__ void stage_padded_a_packet(
    __nv_bfloat16* shared_stage,
    const __nv_bfloat16* sorted_middle,
    int block_m_start,
    int packet,
    int in_features,
    int num_groups) {
  // One uint32 is one BF16 pair.  Padding is initialized once in both
  // persistent buffers; the generic path and tail stage only real D6 pairs.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group_in_packet = lane >> 2;
  const int pair_in_group = lane & 3;
  if constexpr (FullPacket) {
    const int safe_pair = pair_in_group < 3 ? pair_in_group : 2;
    const int group = packet * kGroupsPerPacket + group_in_packet;
    const int shared_k = group_in_packet * kPaddedGroupSize +
        pair_in_group * 2;
    const int real_k = group * kGroupSize + safe_pair * 2;
    const int source_bytes = pair_in_group < 3 ? 4 : 0;
    for (int local_m = warp; local_m < BlockM; local_m += NumWarps) {
      uint32_t* destination = reinterpret_cast<uint32_t*>(
          shared_stage + swizzled_a_offset(local_m, shared_k));
      const __nv_bfloat16* row = sorted_middle +
          static_cast<int64_t>(block_m_start + local_m) * in_features;
      cp_async_ca_4(destination, row + real_k, source_bytes);
    }
  } else if (pair_in_group < 3) {
    const int group = packet * kGroupsPerPacket + group_in_packet;
    const int shared_k = group_in_packet * kPaddedGroupSize +
        pair_in_group * 2;
    const int real_k = group * kGroupSize + pair_in_group * 2;
    const bool valid = group < num_groups && real_k + 1 < in_features;
    for (int local_m = warp; local_m < BlockM; local_m += NumWarps) {
      uint32_t* destination = reinterpret_cast<uint32_t*>(
          shared_stage + swizzled_a_offset(local_m, shared_k));
      const __nv_bfloat16* row = sorted_middle +
          static_cast<int64_t>(block_m_start + local_m) * in_features;
      cp_async_ca_4(destination, valid ? row + real_k : row, valid ? 4 : 0);
    }
  }
  cp_async_commit();
}

template <int BlockM, int NumWarps>
__device__ __forceinline__ void stage_exact_a_packet(
    __nv_bfloat16* shared_stage,
    const __nv_bfloat16* sorted_middle,
    int block_m_start,
    int packet,
    int in_features) {
  // Eight D6 codewords are exactly 48 BF16 values.  Lanes 0..23 each stage
  // one contiguous pair into shared K[0:48]; no D8 padding is constructed.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane < kExactPacketK / 2) {
    const int packet_k = lane * 2;
    const int real_k = packet * kExactPacketK + packet_k;
    for (int local_m = warp; local_m < BlockM; local_m += NumWarps) {
      uint32_t* destination = reinterpret_cast<uint32_t*>(
          shared_stage + swizzled_a_offset(local_m, packet_k));
      const __nv_bfloat16* row = sorted_middle +
          static_cast<int64_t>(block_m_start + local_m) * in_features;
      cp_async_ca_4(destination, row + real_k, 4);
    }
  }
  cp_async_commit();
}

template <int BlockM, int NumWarps>
__device__ __forceinline__ void stage_padded_tail_after_exact(
    __nv_bfloat16* shared_stage,
    const __nv_bfloat16* sorted_middle,
    int block_m_start,
    int packet,
    int in_features,
    int num_groups) {
  // An exact-K48 packet uses the old D8 padding columns as real data.  A
  // following padded tail must therefore overwrite both invalid pairs and
  // every D8 padding pair instead of relying on one-time initialization.
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group_in_packet = lane >> 2;
  const int pair_in_group = lane & 3;
  const int safe_pair = pair_in_group < 3 ? pair_in_group : 0;
  const int group = packet * kGroupsPerPacket + group_in_packet;
  const int shared_k = group_in_packet * kPaddedGroupSize +
      pair_in_group * 2;
  const int real_k = group * kGroupSize + safe_pair * 2;
  const bool valid = pair_in_group < 3 && group < num_groups &&
      real_k + 1 < in_features;
  for (int local_m = warp; local_m < BlockM; local_m += NumWarps) {
    uint32_t* destination = reinterpret_cast<uint32_t*>(
        shared_stage + swizzled_a_offset(local_m, shared_k));
    const __nv_bfloat16* row = sorted_middle +
        static_cast<int64_t>(block_m_start + local_m) * in_features;
    cp_async_ca_4(destination, valid ? row + real_k : row, valid ? 4 : 0);
  }
  cp_async_commit();
}

template <bool FullN>
__device__ __forceinline__ void store_route_pair(
    float accumulator_low,
    float accumulator_high,
    float norm_low,
    float norm_high,
    float router,
    int ticket,
    bool valid_ticket,
    int output_n,
    int out_features,
    __nv_bfloat16* route_output) {
  if constexpr (FullN) {
    if (valid_ticket) {
      const float value0 = accumulator_low * norm_low * router;
      const float value1 = accumulator_high * norm_high * router;
      *reinterpret_cast<__nv_bfloat162*>(
          route_output +
          static_cast<int64_t>(ticket) * out_features + output_n) =
          __floats2bfloat162_rn(value0, value1);
    }
  } else if (output_n < out_features && valid_ticket) {
    const float value0 = accumulator_low * norm_low * router;
    if ((out_features & 1) == 0 && output_n + 1 < out_features) {
      const float value1 = accumulator_high * norm_high * router;
      *reinterpret_cast<__nv_bfloat162*>(
          route_output +
          static_cast<int64_t>(ticket) * out_features + output_n) =
          __floats2bfloat162_rn(value0, value1);
    } else {
      route_output[
          static_cast<int64_t>(ticket) * out_features + output_n] =
          from_float(value0);
      if (output_n + 1 < out_features) {
        const float value1 = accumulator_high * norm_high * router;
        route_output[
            static_cast<int64_t>(ticket) * out_features + output_n + 1] =
            from_float(value1);
      }
    }
  }
}

template <
    int BlockM,
    int NumWarps,
    bool FastShape,
    bool ExactK48,
    bool CoalescedK48>
__global__ __launch_bounds__(NumWarps * kWarpSize)
void down_padded64_control_kernel(
    const __nv_bfloat16* __restrict__ sorted_middle,
    const __nv_bfloat16* __restrict__ codebook,
    const int32_t* __restrict__ assignments,
    const __nv_bfloat16* __restrict__ output_norm,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_tickets,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tickets_post_padded,
    __nv_bfloat16* __restrict__ route_output,
    int num_routes,
    int num_m_blocks,
    int out_features,
    int in_features,
    int codebook_size,
    int codebook_banks,
    int num_groups,
    int num_words) {
  constexpr int kMTiles = BlockM / 16;
  // A 3-D launch encodes Triton's grouped order directly:
  //   x = M block within an 8-block group, y = N block, z = M group.
  // This avoids per-CTA runtime division/modulo while preserving locality.
  const int block_m_index = blockIdx.z * kGroupScheduleM + blockIdx.x;
  const int block_n_index = blockIdx.y;
  if (block_m_index >= num_m_blocks) {
    return;
  }
  const int block_m_start = block_m_index * BlockM;
  if (block_m_start >= *num_tickets_post_padded) {
    return;
  }

  const int expert = expert_ids[block_m_index];
  const int codebook_expert = codebook_banks == 1 ? 0 : expert;
  const auto* expert_codebook = codebook +
      static_cast<int64_t>(codebook_expert) * codebook_size * kGroupSize;
  const auto* expert_assignments = assignments +
      static_cast<int64_t>(expert) * num_words * out_features;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  constexpr int kBlockN = NumWarps * kWarpOutputN;
  const int block_n_start = block_n_index * kBlockN;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);

  float accumulators[kMTiles][2][4];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        accumulators[m][n][value] = 0.0f;
      }
    }
  }

  const int packets = (num_groups + kGroupsPerPacket - 1) /
      kGroupsPerPacket;
  const int full_packets = in_features /
      (kGroupsPerPacket * kGroupSize);
  // Padded packets reuse their two shared-memory buffers, so their D8 padding
  // lanes are zeroed once before the pipeline starts.  Exact K48 packets load
  // only K[0:48], and their padded tail overwrites every K[0:64] pair itself;
  // initializing the unused exact-path padding would therefore be dead work.
  if constexpr (!ExactK48) {
    initialize_padded_a_buffers<BlockM, NumWarps>(shared_a);
  }
  if constexpr (ExactK48) {
    if (full_packets > 0) {
      stage_exact_a_packet<BlockM, NumWarps>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features);
    } else {
      stage_padded_tail_after_exact<BlockM, NumWarps>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    }
  } else if constexpr (FastShape) {
    if (full_packets > 0) {
      stage_padded_a_packet<BlockM, NumWarps, true>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    } else {
      stage_padded_a_packet<BlockM, NumWarps, false>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    }
  } else {
    stage_padded_a_packet<BlockM, NumWarps, false>(
        shared_a,
        sorted_middle,
        block_m_start,
        0,
        in_features,
        num_groups);
  }
  cp_async_wait_all();
  __syncthreads();
  int read_pipe = 0;
  for (int packet = 0; packet < packets; ++packet) {
    const int next_packet = packet + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_packet < packets) {
      if constexpr (ExactK48) {
        if (next_packet < full_packets) {
          stage_exact_a_packet<BlockM, NumWarps>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features);
        } else {
          stage_padded_tail_after_exact<BlockM, NumWarps>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        }
      } else if constexpr (FastShape) {
        if (next_packet < full_packets) {
          stage_padded_a_packet<BlockM, NumWarps, true>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        } else {
          stage_padded_a_packet<BlockM, NumWarps, false>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        }
      } else {
        stage_padded_a_packet<BlockM, NumWarps, false>(
            shared_a + write_pipe * BlockM * kPhysicalPacketK,
            sorted_middle,
            block_m_start,
            next_packet,
            in_features,
            num_groups);
      }
    }
    const auto* shared_stage =
        shared_a + read_pipe * BlockM * kPhysicalPacketK;
    if constexpr (ExactK48) {
      if (packet < full_packets) {
        if constexpr (CoalescedK48) {
          compute_coalesced_k48_packet<BlockM>(
              accumulators,
              shared_stage,
              expert_codebook,
              expert_assignments,
              packet,
              block_n_start,
              out_features,
              out_features,
              1);
        } else {
          compute_exact_k48_packet<BlockM>(
              accumulators,
              shared_stage,
              expert_codebook,
              expert_assignments,
              packet,
              block_n_start,
              out_features,
              out_features,
              1);
        }
      } else {
        compute_padded_packet<BlockM, false>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      }
    } else if constexpr (FastShape) {
      if (packet < full_packets) {
        compute_padded_packet<BlockM, true>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      } else {
        compute_padded_packet<BlockM, false>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      }
    } else {
      compute_padded_packet<BlockM, false>(
          accumulators,
          shared_stage,
          expert_codebook,
          expert_assignments,
          packet,
          block_n_start,
          out_features,
          codebook_size,
          num_groups,
          num_words,
          out_features,
          1);
    }
    if (next_packet < packets) {
      cp_async_wait_all();
      __syncthreads();
    }
    read_pipe = write_pipe;
  }

  float norm_low[2] = {0.0f, 0.0f};
  float norm_high[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    if constexpr (FastShape) {
      norm_low[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n]);
      norm_high[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n + 1]);
    } else if (output_n < out_features) {
      norm_low[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n]);
      if (output_n + 1 < out_features) {
        norm_high[n_half] = __bfloat162float(output_norm[
            static_cast<int64_t>(expert) * out_features + output_n + 1]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    const int row0 = block_m_start + m * 16 + lane_group;
    const int row1 = row0 + 8;
    const int ticket0 = sorted_tickets[row0];
    const int ticket1 = sorted_tickets[row1];
    const bool valid_ticket0 = ticket0 < num_routes;
    const bool valid_ticket1 = ticket1 < num_routes;
    const float router0 = valid_ticket0 ? topk_weights[ticket0] : 0.0f;
    const float router1 = valid_ticket1 ? topk_weights[ticket1] : 0.0f;
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      const int output_n = block_n_start + warp * 16 + n_half * 8 +
          lane_in_group * 2;
      store_route_pair<FastShape>(
          accumulators[m][n_half][0],
          accumulators[m][n_half][1],
          norm_low[n_half],
          norm_high[n_half],
          router0,
          ticket0,
          valid_ticket0,
          output_n,
          out_features,
          route_output);
      store_route_pair<FastShape>(
          accumulators[m][n_half][2],
          accumulators[m][n_half][3],
          norm_low[n_half],
          norm_high[n_half],
          router1,
          ticket1,
          valid_ticket1,
          output_n,
          out_features,
          route_output);
    }
  }
}

template <
    int BlockM,
    int NumWarps,
    bool FastShape,
    bool ExactK48,
    bool CoalescedK48>
__global__ __launch_bounds__(NumWarps * kWarpSize)
void down_padded64_control_task_kernel(
    const __nv_bfloat16* __restrict__ sorted_middle,
    const __nv_bfloat16* __restrict__ codebook,
    const int32_t* __restrict__ assignments,
    const __nv_bfloat16* __restrict__ output_norm,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_tickets,
    __nv_bfloat16* __restrict__ route_output,
    int num_routes,
    int task_capacity,
    int out_features,
    int in_features,
    int codebook_size,
    int codebook_banks,
    int num_groups,
    int num_words,
    const int2* __restrict__ tasks,
    const int32_t* __restrict__ task_counts,
    int task_count_index) {
  constexpr int kMTiles = BlockM / 16;
  const int block_m_index = blockIdx.z * kGroupScheduleM + blockIdx.x;
  const int block_n_index = blockIdx.y;
  if (block_m_index >= task_capacity ||
      block_m_index >= task_counts[task_count_index]) {
    return;
  }
    const int2 task = tasks[block_m_index];
    const int block_m_start = task.x;
    const int expert = task.y;
    const int codebook_expert = codebook_banks == 1 ? 0 : expert;
    const auto* expert_codebook = codebook +
        static_cast<int64_t>(codebook_expert) * codebook_size * kGroupSize;
    const auto* expert_assignments = assignments +
        static_cast<int64_t>(expert) * num_words * out_features;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_group = lane >> 2;
    const int lane_in_group = lane & 3;
    constexpr int kBlockN = NumWarps * kWarpOutputN;
    const int block_n_start = block_n_index * kBlockN;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);

  float accumulators[kMTiles][2][4];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        accumulators[m][n][value] = 0.0f;
      }
    }
  }

  const int packets = (num_groups + kGroupsPerPacket - 1) /
      kGroupsPerPacket;
  const int full_packets = in_features /
      (kGroupsPerPacket * kGroupSize);
  // Padded packets reuse their two shared-memory buffers, so their D8 padding
  // lanes are zeroed once before the pipeline starts.  Exact K48 packets load
  // only K[0:48], and their padded tail overwrites every K[0:64] pair itself;
  // initializing the unused exact-path padding would therefore be dead work.
  if constexpr (!ExactK48) {
    initialize_padded_a_buffers<BlockM, NumWarps>(shared_a);
  }
  if constexpr (ExactK48) {
    if (full_packets > 0) {
      stage_exact_a_packet<BlockM, NumWarps>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features);
    } else {
      stage_padded_tail_after_exact<BlockM, NumWarps>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    }
  } else if constexpr (FastShape) {
    if (full_packets > 0) {
      stage_padded_a_packet<BlockM, NumWarps, true>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    } else {
      stage_padded_a_packet<BlockM, NumWarps, false>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    }
  } else {
    stage_padded_a_packet<BlockM, NumWarps, false>(
        shared_a,
        sorted_middle,
        block_m_start,
        0,
        in_features,
        num_groups);
  }
  cp_async_wait_all();
  __syncthreads();
  int read_pipe = 0;
  for (int packet = 0; packet < packets; ++packet) {
    const int next_packet = packet + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_packet < packets) {
      if constexpr (ExactK48) {
        if (next_packet < full_packets) {
          stage_exact_a_packet<BlockM, NumWarps>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features);
        } else {
          stage_padded_tail_after_exact<BlockM, NumWarps>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        }
      } else if constexpr (FastShape) {
        if (next_packet < full_packets) {
          stage_padded_a_packet<BlockM, NumWarps, true>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        } else {
          stage_padded_a_packet<BlockM, NumWarps, false>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        }
      } else {
        stage_padded_a_packet<BlockM, NumWarps, false>(
            shared_a + write_pipe * BlockM * kPhysicalPacketK,
            sorted_middle,
            block_m_start,
            next_packet,
            in_features,
            num_groups);
      }
    }
    const auto* shared_stage =
        shared_a + read_pipe * BlockM * kPhysicalPacketK;
    if constexpr (ExactK48) {
      if (packet < full_packets) {
        if constexpr (CoalescedK48) {
          compute_coalesced_k48_packet<BlockM>(
              accumulators,
              shared_stage,
              expert_codebook,
              expert_assignments,
              packet,
              block_n_start,
              out_features,
              out_features,
              1);
        } else {
          compute_exact_k48_packet<BlockM>(
              accumulators,
              shared_stage,
              expert_codebook,
              expert_assignments,
              packet,
              block_n_start,
              out_features,
              out_features,
              1);
        }
      } else {
        compute_padded_packet<BlockM, false>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      }
    } else if constexpr (FastShape) {
      if (packet < full_packets) {
        compute_padded_packet<BlockM, true>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      } else {
        compute_padded_packet<BlockM, false>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      }
    } else {
      compute_padded_packet<BlockM, false>(
          accumulators,
          shared_stage,
          expert_codebook,
          expert_assignments,
          packet,
          block_n_start,
          out_features,
          codebook_size,
          num_groups,
          num_words,
          out_features,
          1);
    }
    if (next_packet < packets) {
      cp_async_wait_all();
      __syncthreads();
    }
    read_pipe = write_pipe;
  }

  float norm_low[2] = {0.0f, 0.0f};
  float norm_high[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    if constexpr (FastShape) {
      norm_low[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n]);
      norm_high[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n + 1]);
    } else if (output_n < out_features) {
      norm_low[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n]);
      if (output_n + 1 < out_features) {
        norm_high[n_half] = __bfloat162float(output_norm[
            static_cast<int64_t>(expert) * out_features + output_n + 1]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    const int row0 = block_m_start + m * 16 + lane_group;
    const int row1 = row0 + 8;
    const int ticket0 = sorted_tickets[row0];
    const int ticket1 = sorted_tickets[row1];
    const bool valid_ticket0 = ticket0 < num_routes;
    const bool valid_ticket1 = ticket1 < num_routes;
    const float router0 = valid_ticket0 ? topk_weights[ticket0] : 0.0f;
    const float router1 = valid_ticket1 ? topk_weights[ticket1] : 0.0f;
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      const int output_n = block_n_start + warp * 16 + n_half * 8 +
          lane_in_group * 2;
      store_route_pair<FastShape>(
          accumulators[m][n_half][0],
          accumulators[m][n_half][1],
          norm_low[n_half],
          norm_high[n_half],
          router0,
          ticket0,
          valid_ticket0,
          output_n,
          out_features,
          route_output);
      store_route_pair<FastShape>(
          accumulators[m][n_half][2],
          accumulators[m][n_half][3],
          norm_low[n_half],
          norm_high[n_half],
          router1,
          ticket1,
          valid_ticket1,
          output_n,
          out_features,
          route_output);
    }
  }
}



__device__ __forceinline__ void down_padded64_control_bm16_task_body(
    const __nv_bfloat16* __restrict__ sorted_middle,
    const __nv_bfloat16* __restrict__ codebook,
    const int32_t* __restrict__ assignments,
    const __nv_bfloat16* __restrict__ output_norm,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_tickets,
    __nv_bfloat16* __restrict__ route_output,
    int num_routes,
    int out_features,
    int in_features,
    int codebook_size,
    int codebook_banks,
    int num_groups,
    int num_words,
    const int2* __restrict__ tasks,
    int block_m_index,
    int block_n_index) {
  constexpr int BlockM = 16;
  constexpr int NumWarps = 8;
  constexpr bool FastShape = true;
  constexpr bool ExactK48 = true;
  constexpr bool CoalescedK48 = true;
  constexpr int kMTiles = 1;
    const int2 task = tasks[block_m_index];
    const int block_m_start = task.x;
    const int expert = task.y;
    const int codebook_expert = codebook_banks == 1 ? 0 : expert;
    const auto* expert_codebook = codebook +
        static_cast<int64_t>(codebook_expert) * codebook_size * kGroupSize;
    const auto* expert_assignments = assignments +
        static_cast<int64_t>(expert) * num_words * out_features;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_group = lane >> 2;
    const int lane_in_group = lane & 3;
    constexpr int kBlockN = NumWarps * kWarpOutputN;
    const int block_n_start = block_n_index * kBlockN;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);

  float accumulators[kMTiles][2][4];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        accumulators[m][n][value] = 0.0f;
      }
    }
  }

  const int packets = (num_groups + kGroupsPerPacket - 1) /
      kGroupsPerPacket;
  const int full_packets = in_features /
      (kGroupsPerPacket * kGroupSize);
  // Padded packets reuse their two shared-memory buffers, so their D8 padding
  // lanes are zeroed once before the pipeline starts.  Exact K48 packets load
  // only K[0:48], and their padded tail overwrites every K[0:64] pair itself;
  // initializing the unused exact-path padding would therefore be dead work.
  if constexpr (!ExactK48) {
    initialize_padded_a_buffers<BlockM, NumWarps>(shared_a);
  }
  if constexpr (ExactK48) {
    if (full_packets > 0) {
      stage_exact_a_packet<BlockM, NumWarps>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features);
    } else {
      stage_padded_tail_after_exact<BlockM, NumWarps>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    }
  } else if constexpr (FastShape) {
    if (full_packets > 0) {
      stage_padded_a_packet<BlockM, NumWarps, true>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    } else {
      stage_padded_a_packet<BlockM, NumWarps, false>(
          shared_a,
          sorted_middle,
          block_m_start,
          0,
          in_features,
          num_groups);
    }
  } else {
    stage_padded_a_packet<BlockM, NumWarps, false>(
        shared_a,
        sorted_middle,
        block_m_start,
        0,
        in_features,
        num_groups);
  }
  cp_async_wait_all();
  __syncthreads();
  int read_pipe = 0;
  for (int packet = 0; packet < packets; ++packet) {
    const int next_packet = packet + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_packet < packets) {
      if constexpr (ExactK48) {
        if (next_packet < full_packets) {
          stage_exact_a_packet<BlockM, NumWarps>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features);
        } else {
          stage_padded_tail_after_exact<BlockM, NumWarps>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        }
      } else if constexpr (FastShape) {
        if (next_packet < full_packets) {
          stage_padded_a_packet<BlockM, NumWarps, true>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        } else {
          stage_padded_a_packet<BlockM, NumWarps, false>(
              shared_a + write_pipe * BlockM * kPhysicalPacketK,
              sorted_middle,
              block_m_start,
              next_packet,
              in_features,
              num_groups);
        }
      } else {
        stage_padded_a_packet<BlockM, NumWarps, false>(
            shared_a + write_pipe * BlockM * kPhysicalPacketK,
            sorted_middle,
            block_m_start,
            next_packet,
            in_features,
            num_groups);
      }
    }
    const auto* shared_stage =
        shared_a + read_pipe * BlockM * kPhysicalPacketK;
    if constexpr (ExactK48) {
      if (packet < full_packets) {
        if constexpr (CoalescedK48) {
          compute_coalesced_k48_packet<BlockM>(
              accumulators,
              shared_stage,
              expert_codebook,
              expert_assignments,
              packet,
              block_n_start,
              out_features,
              out_features,
              1);
        } else {
          compute_exact_k48_packet<BlockM>(
              accumulators,
              shared_stage,
              expert_codebook,
              expert_assignments,
              packet,
              block_n_start,
              out_features,
              out_features,
              1);
        }
      } else {
        compute_padded_packet<BlockM, false>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      }
    } else if constexpr (FastShape) {
      if (packet < full_packets) {
        compute_padded_packet<BlockM, true>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      } else {
        compute_padded_packet<BlockM, false>(
            accumulators,
            shared_stage,
            expert_codebook,
            expert_assignments,
            packet,
            block_n_start,
            out_features,
            codebook_size,
            num_groups,
            num_words,
            out_features,
            1);
      }
    } else {
      compute_padded_packet<BlockM, false>(
          accumulators,
          shared_stage,
          expert_codebook,
          expert_assignments,
          packet,
          block_n_start,
          out_features,
          codebook_size,
          num_groups,
          num_words,
          out_features,
          1);
    }
    if (next_packet < packets) {
      cp_async_wait_all();
      __syncthreads();
    }
    read_pipe = write_pipe;
  }

  float norm_low[2] = {0.0f, 0.0f};
  float norm_high[2] = {0.0f, 0.0f};
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    if constexpr (FastShape) {
      norm_low[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n]);
      norm_high[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n + 1]);
    } else if (output_n < out_features) {
      norm_low[n_half] = __bfloat162float(output_norm[
          static_cast<int64_t>(expert) * out_features + output_n]);
      if (output_n + 1 < out_features) {
        norm_high[n_half] = __bfloat162float(output_norm[
            static_cast<int64_t>(expert) * out_features + output_n + 1]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    const int row0 = block_m_start + m * 16 + lane_group;
    const int row1 = row0 + 8;
    const int ticket0 = sorted_tickets[row0];
    const int ticket1 = sorted_tickets[row1];
    const bool valid_ticket0 = ticket0 < num_routes;
    const bool valid_ticket1 = ticket1 < num_routes;
    const float router0 = valid_ticket0 ? topk_weights[ticket0] : 0.0f;
    const float router1 = valid_ticket1 ? topk_weights[ticket1] : 0.0f;
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      const int output_n = block_n_start + warp * 16 + n_half * 8 +
          lane_in_group * 2;
      store_route_pair<FastShape>(
          accumulators[m][n_half][0],
          accumulators[m][n_half][1],
          norm_low[n_half],
          norm_high[n_half],
          router0,
          ticket0,
          valid_ticket0,
          output_n,
          out_features,
          route_output);
      store_route_pair<FastShape>(
          accumulators[m][n_half][2],
          accumulators[m][n_half][3],
          norm_low[n_half],
          norm_high[n_half],
          router1,
          ticket1,
          valid_ticket1,
          output_n,
          out_features,
          route_output);
    }
  }
}

template <int TasksPerCta>
__global__ __launch_bounds__(8 * kWarpSize)
void down_padded64_control_bm16_task_batch_kernel(
    const __nv_bfloat16* __restrict__ sorted_middle,
    const __nv_bfloat16* __restrict__ codebook,
    const int32_t* __restrict__ assignments,
    const __nv_bfloat16* __restrict__ output_norm,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_tickets,
    __nv_bfloat16* __restrict__ route_output,
    int num_routes,
    int out_features,
    int in_features,
    int codebook_size,
    int codebook_banks,
    int num_groups,
    int num_words,
    const int2* __restrict__ tasks,
    const int32_t* __restrict__ task_counts,
    int task_count_index,
    int task_capacity) {
  static_assert(
      TasksPerCta == 1 || TasksPerCta == 2 || TasksPerCta == 4 ||
          TasksPerCta == 8,
      "unsupported adaptive task batch");
  const int task_limit = task_counts[task_count_index];
  const int block_n_index = static_cast<int>(blockIdx.y);
#define NOWAG_DOWN_RUN_TASK_SLOT(Slot)                                    \
  {                                                                      \
    const int task_index = static_cast<int>(blockIdx.x) +                \
        (Slot) * static_cast<int>(gridDim.x);                            \
    if (task_index < task_capacity && task_index < task_limit) {         \
      down_padded64_control_bm16_task_body(                              \
          sorted_middle, codebook, assignments, output_norm,            \
          topk_weights, sorted_tickets, route_output, num_routes,        \
          out_features, in_features, codebook_size, codebook_banks,      \
          num_groups, num_words, tasks, task_index, block_n_index);      \
    }                                                                    \
  }

  NOWAG_DOWN_RUN_TASK_SLOT(0);
  if constexpr (TasksPerCta >= 2) {
    __syncthreads();
    NOWAG_DOWN_RUN_TASK_SLOT(1);
  }
  if constexpr (TasksPerCta >= 4) {
    __syncthreads();
    NOWAG_DOWN_RUN_TASK_SLOT(2);
    __syncthreads();
    NOWAG_DOWN_RUN_TASK_SLOT(3);
  }
  if constexpr (TasksPerCta >= 8) {
    __syncthreads();
    NOWAG_DOWN_RUN_TASK_SLOT(4);
    __syncthreads();
    NOWAG_DOWN_RUN_TASK_SLOT(5);
    __syncthreads();
    NOWAG_DOWN_RUN_TASK_SLOT(6);
    __syncthreads();
    NOWAG_DOWN_RUN_TASK_SLOT(7);
  }

#undef NOWAG_DOWN_RUN_TASK_SLOT
}

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

template <int BlockM, int NumWarps, bool ExactK48, bool CoalescedK48>
void launch_down(
    torch::Tensor sorted_middle,
    torch::Tensor codebook,
    torch::Tensor packed_assignments,
    torch::Tensor output_norm,
    torch::Tensor topk_weights,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor route_output,
    int num_routes,
    int num_m_blocks,
    cudaStream_t stream) {
  const int out_features = output_norm.size(1);
  const int in_features = sorted_middle.size(1);
  const int num_groups = (in_features + kGroupSize - 1) / kGroupSize;
  const int num_words = (num_groups * 12 + 31) / 32;
  constexpr int kBlockN = NumWarps * kWarpOutputN;
  constexpr int kThreads = NumWarps * kWarpSize;
  const int num_n_blocks = (out_features + kBlockN - 1) / kBlockN;
  const dim3 grid(
      kGroupScheduleM,
      num_n_blocks,
      (num_m_blocks + kGroupScheduleM - 1) / kGroupScheduleM);
  constexpr int kSharedBytes =
      2 * BlockM * kPhysicalPacketK * sizeof(__nv_bfloat16);
  auto launch_variant = [&](auto fast_shape_tag, auto exact_k48_tag) {
    constexpr bool kFastShape = decltype(fast_shape_tag)::value;
    constexpr bool kUseExactK48 = decltype(exact_k48_tag)::value;
    down_padded64_control_kernel<
        BlockM,
        NumWarps,
        kFastShape,
        kUseExactK48,
        CoalescedK48><<<
        grid,
        kThreads,
        kSharedBytes,
        stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(sorted_middle.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr()),
        packed_assignments.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(output_norm.data_ptr()),
        topk_weights.data_ptr<float>(),
        sorted_tickets.data_ptr<int32_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tickets_post_padded.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(route_output.data_ptr()),
        num_routes,
        num_m_blocks,
        out_features,
        in_features,
        codebook.size(1),
        codebook.size(0),
        num_groups,
        num_words);
  };
  const bool use_fast_shape =
      out_features % kBlockN == 0 && codebook.size(1) == 4096;
  if constexpr (ExactK48) {
    if (use_fast_shape) {
      launch_variant(std::true_type{}, std::true_type{});
    } else {
      // The first exact implementation specializes the hot serving shape;
      // uncommon N/codebook shapes retain the audited padded fallback.
      launch_variant(std::false_type{}, std::false_type{});
    }
  } else if (use_fast_shape) {
    launch_variant(std::true_type{}, std::false_type{});
  } else {
    launch_variant(std::false_type{}, std::false_type{});
  }
}

template <int BlockM>
void launch_down_tasks(
    torch::Tensor sorted_middle,
    torch::Tensor codebook,
    torch::Tensor packed_assignments,
    torch::Tensor output_norm,
    torch::Tensor topk_weights,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor tasks,
    torch::Tensor task_counts,
    int task_count_index,
    torch::Tensor route_output,
    int num_routes,
    int tasks_per_cta,
    cudaStream_t stream) {
  constexpr int kNumWarps = 8;
  constexpr int kBlockN = kNumWarps * kWarpOutputN;
  constexpr int kThreads = kNumWarps * kWarpSize;
  const int out_features = output_norm.size(1);
  const int in_features = sorted_middle.size(1);
  const int num_groups = (in_features + kGroupSize - 1) / kGroupSize;
  const int num_words = (num_groups * 12 + 31) / 32;
  const int num_n_blocks = out_features / kBlockN;
  const int task_slots = tasks.size(0);
  constexpr int kSharedBytes =
      2 * BlockM * kPhysicalPacketK * sizeof(__nv_bfloat16);
  if constexpr (BlockM == 16) {
    const auto launch_task_batch = [&](auto batch_size) {
      constexpr int kTasksPerCta = decltype(batch_size)::value;
      const int task_grid_slots = std::max(
          1, (task_slots + kTasksPerCta - 1) / kTasksPerCta);
      const dim3 grid(task_grid_slots, num_n_blocks);
      down_padded64_control_bm16_task_batch_kernel<kTasksPerCta><<<
          grid,
          kThreads,
          kSharedBytes,
          stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(sorted_middle.data_ptr()),
          reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr()),
          packed_assignments.data_ptr<int32_t>(),
          reinterpret_cast<const __nv_bfloat16*>(output_norm.data_ptr()),
          topk_weights.data_ptr<float>(),
          sorted_tickets.data_ptr<int32_t>(),
          reinterpret_cast<__nv_bfloat16*>(route_output.data_ptr()),
          num_routes,
          out_features,
          in_features,
          codebook.size(1),
          codebook.size(0),
          num_groups,
          num_words,
          reinterpret_cast<const int2*>(tasks.data_ptr<int32_t>()),
          task_counts.data_ptr<int32_t>(),
          task_count_index,
          task_slots);
    };
    switch (tasks_per_cta) {
      case 1:
        launch_task_batch(std::integral_constant<int, 1>{});
        break;
      case 2:
        launch_task_batch(std::integral_constant<int, 2>{});
        break;
      case 4:
        launch_task_batch(std::integral_constant<int, 4>{});
        break;
      case 8:
        launch_task_batch(std::integral_constant<int, 8>{});
        break;
    }
  } else {
    const dim3 grid(
        kGroupScheduleM,
        num_n_blocks,
        (task_slots + kGroupScheduleM - 1) / kGroupScheduleM);
    down_padded64_control_task_kernel<
        BlockM, kNumWarps, true, true, true><<<
        grid,
        kThreads,
        kSharedBytes,
        stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(sorted_middle.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(codebook.data_ptr()),
        packed_assignments.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16*>(output_norm.data_ptr()),
        topk_weights.data_ptr<float>(),
        sorted_tickets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(route_output.data_ptr()),
        num_routes,
        task_slots,
        out_features,
        in_features,
        codebook.size(1),
        codebook.size(0),
        num_groups,
        num_words,
        reinterpret_cast<const int2*>(tasks.data_ptr<int32_t>()),
        task_counts.data_ptr<int32_t>(),
        task_count_index);
  }
}

void check_and_launch_down(
    torch::Tensor sorted_middle,
    torch::Tensor codebook,
    torch::Tensor packed_assignments,
    torch::Tensor input_norm,
    torch::Tensor output_norm,
    torch::Tensor topk_weights,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor route_output,
    int64_t num_routes,
    int64_t num_m_blocks,
    int64_t block_m,
    int64_t block_n,
    bool use_exact_k48,
    bool use_coalesced_k48,
    int64_t logical_in_features,
    int64_t input_group_start_lane,
    bool preapplied_input_norm) {
  TORCH_CHECK(
      block_m == 16 || block_m == 64 || block_m == 128,
      "block_m must be 16, 64, or 128");
  TORCH_CHECK(block_n == 64 || block_n == 128, "block_n must be 64 or 128");
  TORCH_CHECK(num_routes > 0 && num_m_blocks > 0, "route sizes must be positive");
  check_tensor(sorted_middle, "sorted_middle", at::kBFloat16);
  check_tensor(codebook, "codebook", at::kBFloat16);
  check_tensor(packed_assignments, "packed_assignments", at::kInt);
  if (!preapplied_input_norm) {
    check_tensor(input_norm, "input_norm", at::kBFloat16);
  }
  check_tensor(output_norm, "output_norm", at::kBFloat16);
  check_tensor(topk_weights, "topk_weights", at::kFloat);
  check_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_tensor(expert_ids, "expert_ids", at::kInt);
  check_tensor(num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  check_tensor(route_output, "route_output", at::kBFloat16);
  const auto device = sorted_middle.device();
  const auto check_same_device = [&](const torch::Tensor& tensor,
                                     const char* name) {
    TORCH_CHECK(
        tensor.device() == device,
        name,
        " must be on the same CUDA device as sorted_middle");
  };
  check_same_device(codebook, "codebook");
  check_same_device(packed_assignments, "packed_assignments");
  if (!preapplied_input_norm) {
    check_same_device(input_norm, "input_norm");
  }
  check_same_device(output_norm, "output_norm");
  check_same_device(topk_weights, "topk_weights");
  check_same_device(sorted_tickets, "sorted_tickets");
  check_same_device(expert_ids, "expert_ids");
  check_same_device(num_tickets_post_padded, "num_tickets_post_padded");
  check_same_device(route_output, "route_output");
  TORCH_CHECK(sorted_middle.dim() == 2, "sorted_middle must be [M,K]");
  TORCH_CHECK(sorted_middle.size(1) > 0 && sorted_middle.size(1) % 2 == 0,
              "sorted_middle K must be a positive even number");
  TORCH_CHECK(logical_in_features > 0 &&
                  input_group_start_lane >= 0 &&
                  input_group_start_lane < kGroupSize &&
                  input_group_start_lane + logical_in_features <=
                      sorted_middle.size(1),
              "logical Down K and its start lane must fit sorted_middle");
  TORCH_CHECK(codebook.dim() == 3 && codebook.size(2) == kGroupSize,
              "codebook must be [B,C,6]");
  TORCH_CHECK(codebook.size(0) > 0 && codebook.size(1) > 0 &&
                  codebook.size(1) <= (1 << 12),
              "codebook dimensions are incompatible with 12-bit IDs");
  TORCH_CHECK(output_norm.dim() == 2, "output_norm must be [E,N]");
  const int64_t num_experts = output_norm.size(0);
  TORCH_CHECK(num_experts > 0,
              "private Down tensors must contain at least one expert");
  TORCH_CHECK(codebook.size(0) == 1 || codebook.size(0) == num_experts,
              "codebook bank dimension must be 1 or match private expert count ",
              num_experts);
  TORCH_CHECK(output_norm.size(1) > 0,
              "output_norm N must be positive");
  if (!preapplied_input_norm) {
    TORCH_CHECK(
        input_norm.sizes() ==
            torch::IntArrayRef({num_experts, logical_in_features}),
        "input_norm must be private [E,logical_K]");
  }
  TORCH_CHECK(route_output.dim() == 2, "route_output must be [routes,N]");
  TORCH_CHECK(route_output.size(0) == num_routes,
              "route_output route dimension is inconsistent");
  TORCH_CHECK(route_output.size(1) == output_norm.size(1),
              "route_output N is inconsistent");
  TORCH_CHECK(packed_assignments.dim() == 3,
              "packed_assignments must be word-major [E,W,N]");
  TORCH_CHECK(packed_assignments.size(0) == num_experts,
              "assignments must retain the private expert dimension");
  TORCH_CHECK(packed_assignments.size(2) == output_norm.size(1),
              "assignment N dimension is inconsistent");
  const int64_t num_groups =
      (sorted_middle.size(1) + kGroupSize - 1) / kGroupSize;
  const int64_t expected_words = (num_groups * 12 + 31) / 32;
  TORCH_CHECK(packed_assignments.size(1) == expected_words,
              "assignment word dimension is inconsistent");
  TORCH_CHECK(topk_weights.numel() == num_routes,
              "topk_weights must contain one value per route");
  TORCH_CHECK(num_tickets_post_padded.numel() == 1,
              "num_tickets_post_padded must be a scalar tensor");
  TORCH_CHECK(sorted_tickets.dim() == 1 &&
                  sorted_tickets.numel() >= sorted_middle.size(0),
              "sorted_tickets must cover the sorted_middle row capacity");
  // vLLM allocates a conservative alignment workspace; its static capacity
  // need not itself be a multiple of BlockM.  Only complete blocks are ever
  // launchable, and the device-side post-padded scalar selects the live prefix.
  const int64_t available_m_blocks = sorted_middle.size(0) / block_m;
  const int64_t launch_m_blocks =
      std::min<int64_t>(num_m_blocks, available_m_blocks);
  TORCH_CHECK(launch_m_blocks > 0,
              "aligned route buffers do not contain a complete M block");
  TORCH_CHECK(expert_ids.dim() == 1 &&
                  expert_ids.numel() >= launch_m_blocks,
              "expert_ids is too small for the launchable M blocks");

  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (!preapplied_input_norm) {
    constexpr int kNormThreads = 256;
    const int64_t live_rows = launch_m_blocks * block_m;
    const int64_t norm_elements = live_rows * sorted_middle.size(1);
    const int norm_blocks = static_cast<int>(
        (norm_elements + kNormThreads - 1) / kNormThreads);
    apply_down_input_norm_kernel<<<norm_blocks, kNormThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(sorted_middle.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input_norm.data_ptr()),
        expert_ids.data_ptr<int32_t>(),
        num_tickets_post_padded.data_ptr<int32_t>(),
        live_rows,
        sorted_middle.size(1),
        logical_in_features,
        input_group_start_lane,
        block_m);
  }
  auto launch_tile = [&](auto exact_k48_tag, auto coalesced_k48_tag) {
    constexpr bool kUseExactK48 = decltype(exact_k48_tag)::value;
    constexpr bool kUseCoalescedK48 = decltype(coalesced_k48_tag)::value;
    if (block_m == 16 && block_n == 64) {
      launch_down<16, 4, kUseExactK48, kUseCoalescedK48>(
          sorted_middle, codebook, packed_assignments, output_norm,
          topk_weights, sorted_tickets, expert_ids, num_tickets_post_padded,
          route_output, static_cast<int>(num_routes),
          static_cast<int>(launch_m_blocks), stream);
    } else if (block_m == 16 && block_n == 128) {
      launch_down<16, 8, kUseExactK48, kUseCoalescedK48>(
          sorted_middle, codebook, packed_assignments, output_norm,
          topk_weights, sorted_tickets, expert_ids, num_tickets_post_padded,
          route_output, static_cast<int>(num_routes),
          static_cast<int>(launch_m_blocks), stream);
    } else if (block_m == 64 && block_n == 64) {
      launch_down<64, 4, kUseExactK48, kUseCoalescedK48>(
          sorted_middle, codebook, packed_assignments, output_norm,
          topk_weights, sorted_tickets, expert_ids, num_tickets_post_padded,
          route_output, static_cast<int>(num_routes),
          static_cast<int>(launch_m_blocks), stream);
    } else if (block_m == 64 && block_n == 128) {
      launch_down<64, 8, kUseExactK48, kUseCoalescedK48>(
          sorted_middle, codebook, packed_assignments, output_norm,
          topk_weights, sorted_tickets, expert_ids, num_tickets_post_padded,
          route_output, static_cast<int>(num_routes),
          static_cast<int>(launch_m_blocks), stream);
    } else if (block_m == 128 && block_n == 64) {
      launch_down<128, 4, kUseExactK48, kUseCoalescedK48>(
          sorted_middle, codebook, packed_assignments, output_norm,
          topk_weights, sorted_tickets, expert_ids, num_tickets_post_padded,
          route_output, static_cast<int>(num_routes),
          static_cast<int>(launch_m_blocks), stream);
    } else {
      launch_down<128, 8, kUseExactK48, kUseCoalescedK48>(
          sorted_middle, codebook, packed_assignments, output_norm,
          topk_weights, sorted_tickets, expert_ids, num_tickets_post_padded,
          route_output, static_cast<int>(num_routes),
          static_cast<int>(launch_m_blocks), stream);
    }
  };
  TORCH_CHECK(!use_coalesced_k48 || use_exact_k48,
              "coalesced K48 requires exact K48");
  if (use_coalesced_k48) {
    launch_tile(std::true_type{}, std::true_type{});
  } else if (use_exact_k48) {
    launch_tile(std::true_type{}, std::false_type{});
  } else {
    launch_tile(std::false_type{}, std::false_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void nowag_moe_down_exact_k48_adaptive_cuda(
    torch::Tensor sorted_middle,
    torch::Tensor codebook,
    torch::Tensor packed_assignments,
    torch::Tensor input_norm,
    torch::Tensor output_norm,
    torch::Tensor topk_weights,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor tasks64,
    torch::Tensor tasks16,
    torch::Tensor task_counts,
    torch::Tensor route_output,
    int64_t num_routes,
    int64_t tasks_per_cta,
    int64_t logical_in_features,
    int64_t input_group_start_lane,
    bool preapplied_input_norm) {
  TORCH_CHECK(num_routes > 0, "num_routes must be positive");
  TORCH_CHECK(
      tasks_per_cta == 1 || tasks_per_cta == 2 || tasks_per_cta == 4 ||
          tasks_per_cta == 8,
      "tasks_per_cta must be 1, 2, 4, or 8");
  check_tensor(sorted_middle, "sorted_middle", at::kBFloat16);
  check_tensor(codebook, "codebook", at::kBFloat16);
  check_tensor(packed_assignments, "packed_assignments", at::kInt);
  if (!preapplied_input_norm) {
    check_tensor(input_norm, "input_norm", at::kBFloat16);
  }
  check_tensor(output_norm, "output_norm", at::kBFloat16);
  check_tensor(topk_weights, "topk_weights", at::kFloat);
  check_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_tensor(expert_ids16, "expert_ids16", at::kInt);
  check_tensor(
      num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  check_tensor(tasks64, "tasks64", at::kInt);
  check_tensor(tasks16, "tasks16", at::kInt);
  check_tensor(task_counts, "task_counts", at::kInt);
  check_tensor(route_output, "route_output", at::kBFloat16);

  TORCH_CHECK(sorted_middle.dim() == 2 && sorted_middle.size(0) >= 16 &&
                  sorted_middle.size(1) > 0 &&
                  sorted_middle.size(1) % 2 == 0,
              "sorted_middle must hold a BM16 tile and have positive even K");
  TORCH_CHECK(logical_in_features > 0 &&
                  input_group_start_lane >= 0 &&
                  input_group_start_lane < kGroupSize &&
                  input_group_start_lane + logical_in_features <=
                      sorted_middle.size(1),
              "logical Down K and its start lane must fit sorted_middle");
  TORCH_CHECK(codebook.dim() == 3 && codebook.size(2) == kGroupSize,
              "codebook must be [B,C,6]");
  TORCH_CHECK(codebook.size(0) > 0 && codebook.size(1) == 4096,
              "adaptive Down V1 requires a 4096-entry codebook");
  TORCH_CHECK(output_norm.dim() == 2 && output_norm.size(0) > 0 &&
                  output_norm.size(1) > 0 &&
                  output_norm.size(1) % 128 == 0,
              "adaptive Down V1 requires private [E,N] norms with N "
              "divisible by 128");
  const int64_t num_experts = output_norm.size(0);
  const int64_t out_features = output_norm.size(1);
  TORCH_CHECK(codebook.size(0) == 1 || codebook.size(0) == num_experts,
              "codebook bank dimension must be 1 or match private expert count ",
              num_experts);
  if (!preapplied_input_norm) {
    TORCH_CHECK(
        input_norm.sizes() ==
            torch::IntArrayRef({num_experts, logical_in_features}),
        "input_norm must be private [E,logical_K]");
  }
  TORCH_CHECK(packed_assignments.dim() == 3 &&
                  packed_assignments.size(0) == num_experts &&
                  packed_assignments.size(2) == out_features,
              "packed_assignments must retain private word-major [E,W,N]");
  const int64_t num_groups =
      (sorted_middle.size(1) + kGroupSize - 1) / kGroupSize;
  const int64_t expected_words = (num_groups * 12 + 31) / 32;
  TORCH_CHECK(packed_assignments.size(1) == expected_words,
              "assignment word dimension is inconsistent");
  TORCH_CHECK(topk_weights.numel() == num_routes,
              "topk_weights must contain one value per route");
  TORCH_CHECK(sorted_tickets.dim() == 1 &&
                  sorted_tickets.numel() >= sorted_middle.size(0),
              "sorted_tickets must cover the sorted_middle row capacity");
  TORCH_CHECK(expert_ids16.dim() == 1 &&
                  expert_ids16.numel() >= sorted_middle.size(0) / 16,
              "expert_ids16 must cover every BM16 middle block");
  TORCH_CHECK(num_tickets_post_padded.numel() == 1,
              "num_tickets_post_padded must be scalar");
  TORCH_CHECK(tasks64.dim() == 2 && tasks64.size(1) == 2,
              "tasks64 must be int32[capacity64,2]");
  TORCH_CHECK(tasks16.dim() == 2 && tasks16.size(1) == 2,
              "tasks16 must be an int32 residual-task queue [capacity16,2]");
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
  TORCH_CHECK(route_output.dim() == 2 &&
                  route_output.size(0) == num_routes &&
                  route_output.size(1) == out_features,
              "route_output must be [num_routes,N]");

  const auto device = sorted_middle.device();
  const torch::Tensor tensors[] = {
      codebook, packed_assignments, output_norm, topk_weights,
      sorted_tickets, expert_ids16, num_tickets_post_padded, tasks64,
      tasks16, task_counts, route_output};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device,
                "all adaptive Down tensors must share one CUDA device");
  }
  if (!preapplied_input_norm) {
    TORCH_CHECK(input_norm.device() == device,
                "input_norm must share the adaptive Down CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (!preapplied_input_norm) {
    constexpr int kNormThreads = 256;
    const int64_t norm_elements = sorted_middle.numel();
    const int norm_blocks = static_cast<int>(
        (norm_elements + kNormThreads - 1) / kNormThreads);
    apply_down_input_norm_kernel<<<norm_blocks, kNormThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(sorted_middle.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input_norm.data_ptr()),
        expert_ids16.data_ptr<int32_t>(),
        num_tickets_post_padded.data_ptr<int32_t>(),
        sorted_middle.size(0),
        sorted_middle.size(1),
        logical_in_features,
        input_group_start_lane,
        16);
  }
  launch_down_tasks<64>(
      sorted_middle,
      codebook,
      packed_assignments,
      output_norm,
      topk_weights,
      sorted_tickets,
      expert_ids16,
      num_tickets_post_padded,
      tasks64,
      task_counts,
      0,
      route_output,
      static_cast<int>(num_routes),
      1,
      stream);
  launch_down_tasks<16>(
      sorted_middle,
      codebook,
      packed_assignments,
      output_norm,
      topk_weights,
      sorted_tickets,
      expert_ids16,
      num_tickets_post_padded,
      tasks16,
      task_counts,
      1,
      route_output,
      static_cast<int>(num_routes),
      static_cast<int>(tasks_per_cta),
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nowag_moe_down_padded64_cuda(
    torch::Tensor sorted_middle,
    torch::Tensor codebook,
    torch::Tensor packed_assignments,
    torch::Tensor input_norm,
    torch::Tensor output_norm,
    torch::Tensor topk_weights,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor route_output,
    int64_t num_routes,
    int64_t num_m_blocks,
    int64_t block_m,
    int64_t block_n,
    bool use_exact_k48,
    bool use_coalesced_k48,
    int64_t logical_in_features,
    int64_t input_group_start_lane,
    bool preapplied_input_norm) {
  check_and_launch_down(
      sorted_middle,
      codebook,
      packed_assignments,
      input_norm,
      output_norm,
      topk_weights,
      sorted_tickets,
      expert_ids,
      num_tickets_post_padded,
      route_output,
      num_routes,
      num_m_blocks,
      block_m,
      block_n,
      use_exact_k48,
      use_coalesced_k48,
      logical_in_features,
      input_group_start_lane,
      preapplied_input_norm);
}
