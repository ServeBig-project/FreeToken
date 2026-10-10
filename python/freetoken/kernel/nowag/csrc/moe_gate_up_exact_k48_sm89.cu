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

namespace core = nowag::k48_core;

constexpr int kWarpSize = 32;

struct GateUpProjectionDescriptor {
  const __nv_bfloat16* codebook;
  const int32_t* assignments;
  const __nv_bfloat16* input_norm;
  const __nv_bfloat16* output_norm;
  __nv_bfloat16* output;
  int codebook_size;
  int codebook_banks;
};

struct GateUpProjectionBatch {
  GateUpProjectionDescriptor projections[2];
  int count;
};

__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ void cp_async_ca_16(
    void* shared_destination,
    const void* global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t shared_address = static_cast<uint32_t>(
      __cvta_generic_to_shared(shared_destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16;\n" ::
          "r"(shared_address),
      "l"(global_source));
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      *reinterpret_cast<const uint4*>(global_source);
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

__device__ __forceinline__ uint32_t load_assignment_word_cg(
    const int32_t* assignments,
    int word,
    int output_n,
    int assignment_word_stride,
    int assignment_output_stride) {
  const auto* address = assignments +
      static_cast<int64_t>(word) * assignment_word_stride +
      static_cast<int64_t>(output_n) * assignment_output_stride;
  uint32_t value;
#if defined(__CUDA_ARCH__)
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(address));
#else
  value = static_cast<uint32_t>(*address);
#endif
  return value;
}

template <int SharedEntries>
__device__ __forceinline__ void stage_codebook_prefix(
    __nv_bfloat16* shared_codebook,
    const __nv_bfloat16* global_codebook) {
  static_assert(
      SharedEntries == 0 || SharedEntries == 4096);
  if constexpr (SharedEntries == 0) {
    return;
  }
  constexpr int kPrefixBytes =
      SharedEntries * core::kGroupSize * sizeof(__nv_bfloat16);
  static_assert(kPrefixBytes % 16 == 0);
  constexpr int kChunks = kPrefixBytes / 16;
  constexpr int kCopiesPerGroup = 8;
  auto* shared_bytes = reinterpret_cast<unsigned char*>(shared_codebook);
  const auto* global_bytes =
      reinterpret_cast<const unsigned char*>(global_codebook);
  int copies_in_group = 0;
  for (int chunk = threadIdx.x; chunk < kChunks; chunk += blockDim.x) {
    const int byte_offset = chunk * 16;
    cp_async_ca_16(
        shared_bytes + byte_offset, global_bytes + byte_offset);
    ++copies_in_group;
    if (copies_in_group == kCopiesPerGroup) {
      cp_async_commit();
      copies_in_group = 0;
    }
  }
  if (copies_in_group != 0) {
    cp_async_commit();
  }
}

template <int SharedEntries>
__device__ __forceinline__ uint32_t load_codebook_pair_cached(
    const __nv_bfloat16* global_codebook,
    const __nv_bfloat16* shared_codebook,
    uint32_t id,
    int pair) {
  if constexpr (SharedEntries > 0) {
    if (id < SharedEntries) {
      const auto* address = shared_codebook +
          static_cast<int64_t>(id) * core::kGroupSize + pair;
      const uint32_t shared_address = static_cast<uint32_t>(
          __cvta_generic_to_shared(address));
      uint32_t value;
#if defined(__CUDA_ARCH__)
      asm volatile(
          "ld.shared.u32 %0, [%1];" : "=r"(value) : "r"(shared_address));
#else
      value = *reinterpret_cast<const uint32_t*>(address);
#endif
      return value;
    }
  }
  const auto* address = reinterpret_cast<const uint32_t*>(
      global_codebook + static_cast<int64_t>(id) * core::kGroupSize + pair);
  return __ldg(address);
}

template <int Group, int SharedEntries>
__device__ __forceinline__ uint32_t load_codeword_lane_pair_cached(
    uint32_t word0,
    uint32_t word1,
    uint32_t word2,
    int lane_in_group,
    const __nv_bfloat16* global_codebook,
    const __nv_bfloat16* shared_codebook) {
  const uint32_t id =
      core::decode_packet_id(word0, word1, word2, Group);
  const int pair_lane = lane_in_group < 3 ? lane_in_group : 2;
  return load_codebook_pair_cached<SharedEntries>(
      global_codebook, shared_codebook, id, pair_lane * 2);
}

template <int BlockM, int SharedEntries>
__device__ __forceinline__ void compute_coalesced_k48_packet_cached(
    float (&accumulators)[BlockM / 16][2][4],
    const __nv_bfloat16* shared_stage,
    const __nv_bfloat16* global_codebook,
    const __nv_bfloat16* shared_codebook,
    const int32_t* expert_assignments,
    int packet,
    int block_n_start,
    int out_features,
    int assignment_word_stride,
    int assignment_output_stride) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;
  uint32_t owned_words[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_group;
    const int word_lane = lane_in_group < 3 ? lane_in_group : 2;
    owned_words[n_half] = output_n < out_features
        ? load_assignment_word_cg(
              expert_assignments,
              packet * 3 + word_lane,
              output_n,
              assignment_word_stride,
              assignment_output_stride)
        : 0u;
  }

  uint32_t packet_words[2][3];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    packet_words[n_half][0] = __shfl_sync(
        0xFFFFFFFFu, owned_words[n_half], 0, 4);
    packet_words[n_half][1] = __shfl_sync(
        0xFFFFFFFFu, owned_words[n_half], 1, 4);
    packet_words[n_half][2] = __shfl_sync(
        0xFFFFFFFFu, owned_words[n_half], 2, 4);
  }

  uint32_t carry_group2[2];
  uint32_t b0[2];
  uint32_t b1[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const uint32_t word0 = packet_words[n_half][0];
    const uint32_t word1 = packet_words[n_half][1];
    const uint32_t word2 = packet_words[n_half][2];
    const uint32_t group0 =
        load_codeword_lane_pair_cached<0, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    const uint32_t group1 =
        load_codeword_lane_pair_cached<1, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    b0[n_half] = core::assemble_codeword_operand<0>(
        group0, group1, lane_in_group);
    carry_group2[n_half] =
        load_codeword_lane_pair_cached<2, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    b1[n_half] = core::assemble_codeword_operand<4>(
        group1, carry_group2[n_half], lane_in_group);
  }
  core::mma_exact_fragment<BlockM, 0>(
      accumulators, shared_stage, b0, b1);

  uint32_t carry_group5[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const uint32_t word0 = packet_words[n_half][0];
    const uint32_t word1 = packet_words[n_half][1];
    const uint32_t word2 = packet_words[n_half][2];
    const uint32_t group3 =
        load_codeword_lane_pair_cached<3, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    b0[n_half] = core::assemble_codeword_operand<8>(
        carry_group2[n_half], group3, lane_in_group);
    const uint32_t group4 =
        load_codeword_lane_pair_cached<4, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    carry_group5[n_half] =
        load_codeword_lane_pair_cached<5, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    b1[n_half] = core::assemble_codeword_operand<12>(
        group4, carry_group5[n_half], lane_in_group);
  }
  core::mma_exact_fragment<BlockM, 1>(
      accumulators, shared_stage, b0, b1);

#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const uint32_t word0 = packet_words[n_half][0];
    const uint32_t word1 = packet_words[n_half][1];
    const uint32_t word2 = packet_words[n_half][2];
    const uint32_t group6 =
        load_codeword_lane_pair_cached<6, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    b0[n_half] = core::assemble_codeword_operand<16>(
        carry_group5[n_half], group6, lane_in_group);
    const uint32_t group7 =
        load_codeword_lane_pair_cached<7, SharedEntries>(
            word0,
            word1,
            word2,
            lane_in_group,
            global_codebook,
            shared_codebook);
    b1[n_half] = core::assemble_codeword_operand<20>(
        group6, group7, lane_in_group);
  }
  core::mma_exact_fragment<BlockM, 2>(
      accumulators, shared_stage, b0, b1);
}

template <int BlockM, int SharedEntries>
__device__ __forceinline__ void compute_padded_tail_cached(
    float (&accumulators)[BlockM / 16][2][4],
    const __nv_bfloat16* shared_stage,
    const __nv_bfloat16* global_codebook,
    const __nv_bfloat16* shared_codebook,
    const int32_t* expert_assignments,
    int packet,
    int block_n_start,
    int out_features,
    int codebook_size,
    int num_groups,
    int num_words,
    int assignment_word_stride,
    int assignment_output_stride) {
  constexpr int kMTiles = BlockM / 16;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;

  uint32_t owned_words[2] = {0, 0};
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_group;
    if (lane_in_group < 3 && output_n < out_features) {
      const int word = packet * 3 + lane_in_group;
      if (word < num_words) {
        owned_words[n_half] = load_assignment_word_cg(
            expert_assignments,
            word,
            output_n,
            assignment_word_stride,
            assignment_output_stride);
      }
    }
  }

  uint32_t packet_words[2][3];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    packet_words[n_half][0] = __shfl_sync(
        0xFFFFFFFFu, owned_words[n_half], 0, 4);
    packet_words[n_half][1] = __shfl_sync(
        0xFFFFFFFFu, owned_words[n_half], 1, 4);
    packet_words[n_half][2] = __shfl_sync(
        0xFFFFFFFFu, owned_words[n_half], 2, 4);
  }

#pragma unroll
  for (int fragment = 0; fragment < 4; ++fragment) {
    uint32_t b0_by_half[2] = {0, 0};
    uint32_t b1_by_half[2] = {0, 0};
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      const int output_n = block_n_start + warp * 16 + n_half * 8 +
          lane_group;
      const uint32_t word0 = packet_words[n_half][0];
      const uint32_t word1 = packet_words[n_half][1];
      const uint32_t word2 = packet_words[n_half][2];
      const int group0 = fragment * 2;
      const int group1 = group0 + 1;
      const uint32_t id0 =
          core::decode_packet_id(word0, word1, word2, group0);
      const uint32_t id1 =
          core::decode_packet_id(word0, word1, word2, group1);
      if (output_n < out_features && lane_in_group < 3) {
        const int pair = lane_in_group * 2;
        if (packet * core::kGroupsPerPacket + group0 < num_groups &&
            id0 < codebook_size) {
          b0_by_half[n_half] = load_codebook_pair_cached<SharedEntries>(
              global_codebook, shared_codebook, id0, pair);
        }
        if (packet * core::kGroupsPerPacket + group1 < num_groups &&
            id1 < codebook_size) {
          b1_by_half[n_half] = load_codebook_pair_cached<SharedEntries>(
              global_codebook, shared_codebook, id1, pair);
        }
      }
    }

#pragma unroll
    for (int m = 0; m < kMTiles; ++m) {
      uint32_t a[4];
      core::load_a_fragment(a, shared_stage, m, fragment, lane);
#pragma unroll
      for (int n_half = 0; n_half < 2; ++n_half) {
        core::mma_bf16_m16n8k16(
            accumulators[m][n_half],
            a,
            b0_by_half[n_half],
            b1_by_half[n_half]);
      }
    }
  }
}

template <int BlockM, int NumWarps, bool PrecomputeTokens>
__device__ __forceinline__ void stage_gate_up_exact_a_packet(
    __nv_bfloat16* shared_stage,
    const int32_t* shared_tokens,
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* input_norm,
    const int32_t* sorted_tickets,
    int expert,
    int block_m_start,
    int packet,
    int num_routes,
    int top_k,
    int in_features) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane >= core::kExactPacketK / 2) {
    return;
  }
  const int packet_k = lane * 2;
  const int real_k = packet * core::kExactPacketK + packet_k;
  const auto* expert_norm = input_norm +
      static_cast<int64_t>(expert) * in_features;
  for (int local_m = warp; local_m < BlockM; local_m += NumWarps) {
    int token = -1;
    bool valid_ticket = false;
    if constexpr (PrecomputeTokens) {
      token = shared_tokens[local_m];
      valid_ticket = token >= 0;
    } else {
      const int ticket = sorted_tickets[block_m_start + local_m];
      valid_ticket = ticket >= 0 && ticket < num_routes;
      if (valid_ticket) {
        token = ticket / top_k;
      }
    }
    float value0 = 0.0f;
    float value1 = 0.0f;
    if (valid_ticket) {
      const auto* row = hidden + static_cast<int64_t>(token) * in_features;
      value0 = __bfloat162float(row[real_k]) *
          __bfloat162float(expert_norm[real_k]);
      value1 = __bfloat162float(row[real_k + 1]) *
          __bfloat162float(expert_norm[real_k + 1]);
    }
    auto* destination = shared_stage +
        core::swizzled_a_offset(local_m, packet_k);
    *reinterpret_cast<__nv_bfloat162*>(destination) =
        __floats2bfloat162_rn(value0, value1);
  }
}

template <int BlockM, int NumWarps, bool PrecomputeTokens>
__device__ __forceinline__ void stage_gate_up_padded_tail(
    __nv_bfloat16* shared_stage,
    const int32_t* shared_tokens,
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* input_norm,
    const int32_t* sorted_tickets,
    int expert,
    int block_m_start,
    int packet,
    int num_routes,
    int top_k,
    int in_features,
    int num_groups) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group_in_packet = lane >> 2;
  const int pair_in_group = lane & 3;
  const int group = packet * core::kGroupsPerPacket + group_in_packet;
  const int real_k = group * core::kGroupSize + pair_in_group * 2;
  const int shared_k = group_in_packet * core::kPaddedGroupSize +
      pair_in_group * 2;
  const bool valid_k = pair_in_group < core::kGroupSize / 2 &&
      group < num_groups && real_k + 1 < in_features;
  const auto* expert_norm = input_norm +
      static_cast<int64_t>(expert) * in_features;
  for (int local_m = warp; local_m < BlockM; local_m += NumWarps) {
    int token = -1;
    bool valid_ticket = false;
    if constexpr (PrecomputeTokens) {
      token = shared_tokens[local_m];
      valid_ticket = token >= 0;
    } else {
      const int ticket = sorted_tickets[block_m_start + local_m];
      valid_ticket = ticket >= 0 && ticket < num_routes;
      if (valid_ticket) {
        token = ticket / top_k;
      }
    }
    float value0 = 0.0f;
    float value1 = 0.0f;
    if (valid_ticket && valid_k) {
      const auto* row = hidden + static_cast<int64_t>(token) * in_features;
      value0 = __bfloat162float(row[real_k]) *
          __bfloat162float(expert_norm[real_k]);
      value1 = __bfloat162float(row[real_k + 1]) *
          __bfloat162float(expert_norm[real_k + 1]);
    }
    auto* destination = shared_stage +
        core::swizzled_a_offset(local_m, shared_k);
    *reinterpret_cast<__nv_bfloat162*>(destination) =
        __floats2bfloat162_rn(value0, value1);
  }
}

template <int BlockM, int NumWarps, bool PrecomputeTokens>
__device__ __forceinline__ void stage_gate_up_a_packet(
    __nv_bfloat16* shared_stage,
    const int32_t* shared_tokens,
    const __nv_bfloat16* hidden,
    const __nv_bfloat16* input_norm,
    const int32_t* sorted_tickets,
    int expert,
    int block_m_start,
    int packet,
    int full_packets,
    int num_routes,
    int top_k,
    int in_features,
    int num_groups) {
  if (packet < full_packets) {
    stage_gate_up_exact_a_packet<BlockM, NumWarps, PrecomputeTokens>(
        shared_stage,
        shared_tokens,
        hidden,
        input_norm,
        sorted_tickets,
        expert,
        block_m_start,
        packet,
        num_routes,
        top_k,
        in_features);
  } else {
    stage_gate_up_padded_tail<BlockM, NumWarps, PrecomputeTokens>(
        shared_stage,
        shared_tokens,
        hidden,
        input_norm,
        sorted_tickets,
        expert,
        block_m_start,
        packet,
        num_routes,
        top_k,
        in_features,
        num_groups);
  }
}

template <int BlockM, int NumWarps, bool PrecomputeTokens>
__global__ __launch_bounds__(NumWarps * kWarpSize)
void gate_up_exact_projection_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const int32_t* __restrict__ gate_assignments,
    const __nv_bfloat16* __restrict__ gate_input_norm,
    const __nv_bfloat16* __restrict__ gate_output_norm,
    __nv_bfloat16* __restrict__ gate_output,
    int gate_codebook_size,
    int gate_codebook_banks,
    const __nv_bfloat16* __restrict__ up_codebook,
    const int32_t* __restrict__ up_assignments,
    const __nv_bfloat16* __restrict__ up_input_norm,
    const __nv_bfloat16* __restrict__ up_output_norm,
    __nv_bfloat16* __restrict__ up_output,
    int up_codebook_size,
    int up_codebook_banks,
    const int32_t* __restrict__ sorted_tickets,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tickets_post_padded,
    int projection_count,
    int output_rows,
    int num_routes,
    int top_k,
    int in_features,
    int out_features,
    int output_stride,
    int output_start_lane,
    int num_groups,
    int num_words,
    int alignment_block_ratio) {
  constexpr int kMTiles = BlockM / 16;
  constexpr int kBlockN = NumWarps * 16;
  const int block_m_index = blockIdx.x;
  const int block_m_start = block_m_index * BlockM;
  const int padded_rows = *num_tickets_post_padded;
  if (block_m_start >= padded_rows) {
    return;
  }
  const int tiles_per_projection = out_features / kBlockN;
  const int projection_index = blockIdx.y / tiles_per_projection;
  if (projection_index >= projection_count) {
    return;
  }
  const bool use_up = projection_index != 0;
  const auto* codebook = use_up ? up_codebook : gate_codebook;
  const auto* assignments = use_up ? up_assignments : gate_assignments;
  const auto* input_norm = use_up ? up_input_norm : gate_input_norm;
  const auto* output_norm = use_up ? up_output_norm : gate_output_norm;
  auto* projection_output = use_up ? up_output : gate_output;
  const int codebook_size =
      use_up ? up_codebook_size : gate_codebook_size;
  const int codebook_banks =
      use_up ? up_codebook_banks : gate_codebook_banks;
  const int block_n_start =
      (blockIdx.y - projection_index * tiles_per_projection) * kBlockN;
  const int expert = expert_ids[block_m_index / alignment_block_ratio];
  const int codebook_expert = codebook_banks == 1 ? 0 : expert;
  const auto* expert_codebook = codebook +
      static_cast<int64_t>(codebook_expert) * codebook_size * core::kGroupSize;
  const auto* expert_assignments = assignments +
      static_cast<int64_t>(expert) * num_words * out_features;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);
  auto* shared_tokens = reinterpret_cast<int32_t*>(
      shared_a + 2 * BlockM * core::kPhysicalPacketK);

  // Ticket routing is invariant across all K48 packets in this CTA.  The
  // baseline path reloaded the ticket and recomputed the runtime integer
  // division once per packet.  Build a tiny row table once and let both the
  // exact packets and padded tail consume it.
  if constexpr (PrecomputeTokens) {
    for (int local_m = threadIdx.x; local_m < BlockM;
         local_m += blockDim.x) {
      const int ticket = sorted_tickets[block_m_start + local_m];
      shared_tokens[local_m] =
          ticket >= 0 && ticket < num_routes ? ticket / top_k : -1;
    }
    __syncthreads();
  }

  float accumulators[kMTiles][2][4];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        accumulators[m][n_half][value] = 0.0f;
      }
    }
  }

  const int packets =
      (num_groups + core::kGroupsPerPacket - 1) / core::kGroupsPerPacket;
  const int full_packets = in_features / core::kExactPacketK;
  stage_gate_up_a_packet<BlockM, NumWarps, PrecomputeTokens>(
      shared_a,
      shared_tokens,
      hidden,
      input_norm,
      sorted_tickets,
      expert,
      block_m_start,
      0,
      full_packets,
      num_routes,
      top_k,
      in_features,
      num_groups);
  __syncthreads();

  int read_pipe = 0;
  for (int packet = 0; packet < packets; ++packet) {
    const int next_packet = packet + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_packet < packets) {
      stage_gate_up_a_packet<BlockM, NumWarps, PrecomputeTokens>(
          shared_a + write_pipe * BlockM * core::kPhysicalPacketK,
          shared_tokens,
          hidden,
          input_norm,
          sorted_tickets,
          expert,
          block_m_start,
          next_packet,
          full_packets,
          num_routes,
          top_k,
          in_features,
          num_groups);
    }

    const auto* shared_stage =
        shared_a + read_pipe * BlockM * core::kPhysicalPacketK;
    if (packet < full_packets) {
      core::compute_coalesced_k48_packet<BlockM>(
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
      core::compute_padded_packet<BlockM, false>(
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
      __syncthreads();
    }
    read_pipe = write_pipe;
  }

  float norm_low[2];
  float norm_high[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    norm_low[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n]);
    norm_high[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n + 1]);
  }

#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    const int row0 = block_m_start + m * 16 + lane_group;
    const int row1 = row0 + 8;
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      const int output_n = block_n_start + warp * 16 + n_half * 8 +
          lane_in_group * 2;
      if (row0 < output_rows) {
        auto* destination = projection_output +
            static_cast<int64_t>(row0) * output_stride + output_start_lane +
            output_n;
        const auto values = __floats2bfloat162_rn(
            accumulators[m][n_half][0] * norm_low[n_half],
            accumulators[m][n_half][1] * norm_high[n_half]);
        if ((output_start_lane & 1) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(destination) = values;
        } else {
          destination[0] = __low2bfloat16(values);
          destination[1] = __high2bfloat16(values);
        }
      }
      if (row1 < output_rows) {
        auto* destination = projection_output +
            static_cast<int64_t>(row1) * output_stride + output_start_lane +
            output_n;
        const auto values = __floats2bfloat162_rn(
            accumulators[m][n_half][2] * norm_low[n_half],
            accumulators[m][n_half][3] * norm_high[n_half]);
        if ((output_start_lane & 1) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(destination) = values;
        } else {
          destination[0] = __low2bfloat16(values);
          destination[1] = __high2bfloat16(values);
        }
      }
    }
  }
}

template <
    int BlockM,
    int NumWarps,
    bool PrecomputeTokens>
__global__ __launch_bounds__(NumWarps * kWarpSize)
void gate_up_exact_projection_task_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const int32_t* __restrict__ gate_assignments,
    const __nv_bfloat16* __restrict__ gate_input_norm,
    const __nv_bfloat16* __restrict__ gate_output_norm,
    __nv_bfloat16* __restrict__ gate_output,
    int gate_codebook_size,
    int gate_codebook_banks,
    const __nv_bfloat16* __restrict__ up_codebook,
    const int32_t* __restrict__ up_assignments,
    const __nv_bfloat16* __restrict__ up_input_norm,
    const __nv_bfloat16* __restrict__ up_output_norm,
    __nv_bfloat16* __restrict__ up_output,
    int up_codebook_size,
    int up_codebook_banks,
    const int32_t* __restrict__ sorted_tickets,
    int projection_count,
    int output_rows,
    int num_routes,
    int top_k,
    int in_features,
    int out_features,
    int output_stride,
    int output_start_lane,
    int num_groups,
    int num_words,
    const int2* __restrict__ tasks,
    const int32_t* __restrict__ task_counts,
    int task_count_index) {
  constexpr int kMTiles = BlockM / 16;
  constexpr int kBlockN = NumWarps * 16;
  const int block_m_index = blockIdx.x;
  if (block_m_index >= task_counts[task_count_index]) {
    return;
  }
    const int2 task = tasks[block_m_index];
    const int block_m_start = task.x;
    const int expert = task.y;
    const int tiles_per_projection = out_features / kBlockN;
    const int projection_index = blockIdx.y / tiles_per_projection;
    if (projection_index >= projection_count) {
      return;
    }
    const bool use_up = projection_index != 0;
    const auto* codebook = use_up ? up_codebook : gate_codebook;
    const auto* assignments = use_up ? up_assignments : gate_assignments;
    const auto* input_norm = use_up ? up_input_norm : gate_input_norm;
    const auto* output_norm = use_up ? up_output_norm : gate_output_norm;
    auto* projection_output = use_up ? up_output : gate_output;
    const int codebook_size =
        use_up ? up_codebook_size : gate_codebook_size;
    const int codebook_banks =
        use_up ? up_codebook_banks : gate_codebook_banks;
    const int block_n_start =
        (blockIdx.y - projection_index * tiles_per_projection) * kBlockN;
    const int codebook_expert = codebook_banks == 1 ? 0 : expert;
    const auto* expert_codebook = codebook + static_cast<int64_t>(
        codebook_expert) * codebook_size * core::kGroupSize;
    const auto* expert_assignments = assignments +
        static_cast<int64_t>(expert) * num_words * out_features;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_group = lane >> 2;
    const int lane_in_group = lane & 3;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* shared_tokens = reinterpret_cast<int32_t*>(
        shared_a + 2 * BlockM * core::kPhysicalPacketK);

  // Ticket routing is invariant across all K48 packets in this CTA.  The
  // baseline path reloaded the ticket and recomputed the runtime integer
  // division once per packet.  Build a tiny row table once and let both the
  // exact packets and padded tail consume it.
  if constexpr (PrecomputeTokens) {
    for (int local_m = threadIdx.x; local_m < BlockM;
         local_m += blockDim.x) {
      const int ticket = sorted_tickets[block_m_start + local_m];
      shared_tokens[local_m] =
          ticket >= 0 && ticket < num_routes ? ticket / top_k : -1;
    }
    __syncthreads();
  }

  float accumulators[kMTiles][2][4];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        accumulators[m][n_half][value] = 0.0f;
      }
    }
  }

  const int packets =
      (num_groups + core::kGroupsPerPacket - 1) / core::kGroupsPerPacket;
  const int full_packets = in_features / core::kExactPacketK;
  stage_gate_up_a_packet<BlockM, NumWarps, PrecomputeTokens>(
      shared_a,
      shared_tokens,
      hidden,
      input_norm,
      sorted_tickets,
      expert,
      block_m_start,
      0,
      full_packets,
      num_routes,
      top_k,
      in_features,
      num_groups);
  __syncthreads();

  int read_pipe = 0;
  for (int packet = 0; packet < packets; ++packet) {
    const int next_packet = packet + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_packet < packets) {
      stage_gate_up_a_packet<BlockM, NumWarps, PrecomputeTokens>(
          shared_a + write_pipe * BlockM * core::kPhysicalPacketK,
          shared_tokens,
          hidden,
          input_norm,
          sorted_tickets,
          expert,
          block_m_start,
          next_packet,
          full_packets,
          num_routes,
          top_k,
          in_features,
          num_groups);
    }

    const auto* shared_stage =
        shared_a + read_pipe * BlockM * core::kPhysicalPacketK;
    if (packet < full_packets) {
      core::compute_coalesced_k48_packet<BlockM>(
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
      core::compute_padded_packet<BlockM, false>(
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
      __syncthreads();
    }
    read_pipe = write_pipe;
  }

  float norm_low[2];
  float norm_high[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    norm_low[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n]);
    norm_high[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n + 1]);
  }

#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    const int row0 = block_m_start + m * 16 + lane_group;
    const int row1 = row0 + 8;
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      const int output_n = block_n_start + warp * 16 + n_half * 8 +
          lane_in_group * 2;
      if (row0 < output_rows) {
        auto* destination = projection_output +
            static_cast<int64_t>(row0) * output_stride + output_start_lane +
            output_n;
        const auto values = __floats2bfloat162_rn(
            accumulators[m][n_half][0] * norm_low[n_half],
            accumulators[m][n_half][1] * norm_high[n_half]);
        if ((output_start_lane & 1) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(destination) = values;
        } else {
          destination[0] = __low2bfloat16(values);
          destination[1] = __high2bfloat16(values);
        }
      }
      if (row1 < output_rows) {
        auto* destination = projection_output +
            static_cast<int64_t>(row1) * output_stride + output_start_lane +
            output_n;
        const auto values = __floats2bfloat162_rn(
            accumulators[m][n_half][2] * norm_low[n_half],
            accumulators[m][n_half][3] * norm_high[n_half]);
        if ((output_start_lane & 1) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(destination) = values;
        } else {
          destination[0] = __low2bfloat16(values);
          destination[1] = __high2bfloat16(values);
        }
      }
    }
  }
}



__device__ __forceinline__ void gate_up_exact_projection_bm16_task_body(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const int32_t* __restrict__ gate_assignments,
    const __nv_bfloat16* __restrict__ gate_input_norm,
    const __nv_bfloat16* __restrict__ gate_output_norm,
    __nv_bfloat16* __restrict__ gate_output,
    int gate_codebook_size,
    int gate_codebook_banks,
    const __nv_bfloat16* __restrict__ up_codebook,
    const int32_t* __restrict__ up_assignments,
    const __nv_bfloat16* __restrict__ up_input_norm,
    const __nv_bfloat16* __restrict__ up_output_norm,
    __nv_bfloat16* __restrict__ up_output,
    int up_codebook_size,
    int up_codebook_banks,
    const int32_t* __restrict__ sorted_tickets,
    int projection_count,
    int output_rows,
    int num_routes,
    int top_k,
    int in_features,
    int out_features,
    int output_stride,
    int output_start_lane,
    int num_groups,
    int num_words,
    const int2* __restrict__ tasks,
    int block_m_index) {
  constexpr int BlockM = 16;
  constexpr int NumWarps = 8;
  constexpr bool PrecomputeTokens = true;
  constexpr int kMTiles = 1;
  constexpr int kBlockN = NumWarps * 16;
    const int2 task = tasks[block_m_index];
    const int block_m_start = task.x;
    const int expert = task.y;
    const int tiles_per_projection = out_features / kBlockN;
    const int projection_index = blockIdx.y / tiles_per_projection;
    if (projection_index >= projection_count) {
      return;
    }
    const bool use_up = projection_index != 0;
    const auto* codebook = use_up ? up_codebook : gate_codebook;
    const auto* assignments = use_up ? up_assignments : gate_assignments;
    const auto* input_norm = use_up ? up_input_norm : gate_input_norm;
    const auto* output_norm = use_up ? up_output_norm : gate_output_norm;
    auto* projection_output = use_up ? up_output : gate_output;
    const int codebook_size =
        use_up ? up_codebook_size : gate_codebook_size;
    const int codebook_banks =
        use_up ? up_codebook_banks : gate_codebook_banks;
    const int block_n_start =
        (blockIdx.y - projection_index * tiles_per_projection) * kBlockN;
    const int codebook_expert = codebook_banks == 1 ? 0 : expert;
    const auto* expert_codebook = codebook + static_cast<int64_t>(
        codebook_expert) * codebook_size * core::kGroupSize;
    const auto* expert_assignments = assignments +
        static_cast<int64_t>(expert) * num_words * out_features;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int lane_group = lane >> 2;
    const int lane_in_group = lane & 3;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* shared_tokens = reinterpret_cast<int32_t*>(
        shared_a + 2 * BlockM * core::kPhysicalPacketK);

  // Ticket routing is invariant across all K48 packets in this CTA.  The
  // baseline path reloaded the ticket and recomputed the runtime integer
  // division once per packet.  Build a tiny row table once and let both the
  // exact packets and padded tail consume it.
  if constexpr (PrecomputeTokens) {
    for (int local_m = threadIdx.x; local_m < BlockM;
         local_m += blockDim.x) {
      const int ticket = sorted_tickets[block_m_start + local_m];
      shared_tokens[local_m] =
          ticket >= 0 && ticket < num_routes ? ticket / top_k : -1;
    }
    __syncthreads();
  }

  float accumulators[kMTiles][2][4];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
#pragma unroll
      for (int value = 0; value < 4; ++value) {
        accumulators[m][n_half][value] = 0.0f;
      }
    }
  }

  const int packets =
      (num_groups + core::kGroupsPerPacket - 1) / core::kGroupsPerPacket;
  const int full_packets = in_features / core::kExactPacketK;
  stage_gate_up_a_packet<BlockM, NumWarps, PrecomputeTokens>(
      shared_a,
      shared_tokens,
      hidden,
      input_norm,
      sorted_tickets,
      expert,
      block_m_start,
      0,
      full_packets,
      num_routes,
      top_k,
      in_features,
      num_groups);
  __syncthreads();

  int read_pipe = 0;
  for (int packet = 0; packet < packets; ++packet) {
    const int next_packet = packet + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_packet < packets) {
      stage_gate_up_a_packet<BlockM, NumWarps, PrecomputeTokens>(
          shared_a + write_pipe * BlockM * core::kPhysicalPacketK,
          shared_tokens,
          hidden,
          input_norm,
          sorted_tickets,
          expert,
          block_m_start,
          next_packet,
          full_packets,
          num_routes,
          top_k,
          in_features,
          num_groups);
    }

    const auto* shared_stage =
        shared_a + read_pipe * BlockM * core::kPhysicalPacketK;
    if (packet < full_packets) {
      core::compute_coalesced_k48_packet<BlockM>(
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
      core::compute_padded_packet<BlockM, false>(
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
      __syncthreads();
    }
    read_pipe = write_pipe;
  }

  float norm_low[2];
  float norm_high[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    norm_low[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n]);
    norm_high[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n + 1]);
  }

#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    const int row0 = block_m_start + m * 16 + lane_group;
    const int row1 = row0 + 8;
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      const int output_n = block_n_start + warp * 16 + n_half * 8 +
          lane_in_group * 2;
      if (row0 < output_rows) {
        auto* destination = projection_output +
            static_cast<int64_t>(row0) * output_stride + output_start_lane +
            output_n;
        const auto values = __floats2bfloat162_rn(
            accumulators[m][n_half][0] * norm_low[n_half],
            accumulators[m][n_half][1] * norm_high[n_half]);
        if ((output_start_lane & 1) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(destination) = values;
        } else {
          destination[0] = __low2bfloat16(values);
          destination[1] = __high2bfloat16(values);
        }
      }
      if (row1 < output_rows) {
        auto* destination = projection_output +
            static_cast<int64_t>(row1) * output_stride + output_start_lane +
            output_n;
        const auto values = __floats2bfloat162_rn(
            accumulators[m][n_half][2] * norm_low[n_half],
            accumulators[m][n_half][3] * norm_high[n_half]);
        if ((output_start_lane & 1) == 0) {
          *reinterpret_cast<__nv_bfloat162*>(destination) = values;
        } else {
          destination[0] = __low2bfloat16(values);
          destination[1] = __high2bfloat16(values);
        }
      }
    }
  }
}

template <int TasksPerCta>
__global__ __launch_bounds__(8 * kWarpSize)
void gate_up_exact_projection_bm16_task_batch_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const int32_t* __restrict__ gate_assignments,
    const __nv_bfloat16* __restrict__ gate_input_norm,
    const __nv_bfloat16* __restrict__ gate_output_norm,
    __nv_bfloat16* __restrict__ gate_output,
    int gate_codebook_size,
    int gate_codebook_banks,
    const __nv_bfloat16* __restrict__ up_codebook,
    const int32_t* __restrict__ up_assignments,
    const __nv_bfloat16* __restrict__ up_input_norm,
    const __nv_bfloat16* __restrict__ up_output_norm,
    __nv_bfloat16* __restrict__ up_output,
    int up_codebook_size,
    int up_codebook_banks,
    const int32_t* __restrict__ sorted_tickets,
    int projection_count,
    int output_rows,
    int num_routes,
    int top_k,
    int in_features,
    int out_features,
    int output_stride,
    int output_start_lane,
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
#define NOWAG_GATE_UP_RUN_TASK_SLOT(Slot)                                  \
  {                                                                        \
    const int task_index = static_cast<int>(blockIdx.x) +                  \
        (Slot) * static_cast<int>(gridDim.x);                              \
    if (task_index < task_capacity && task_index < task_limit) {           \
      gate_up_exact_projection_bm16_task_body(                             \
          hidden, gate_codebook, gate_assignments, gate_input_norm,        \
          gate_output_norm, gate_output, gate_codebook_size,               \
          gate_codebook_banks, up_codebook, up_assignments, up_input_norm, \
          up_output_norm, up_output, up_codebook_size, up_codebook_banks,  \
          sorted_tickets, projection_count, output_rows, num_routes,       \
          top_k, in_features, out_features, output_stride,                 \
          output_start_lane, num_groups, num_words, tasks, task_index);    \
    }                                                                      \
  }

  NOWAG_GATE_UP_RUN_TASK_SLOT(0);
  if constexpr (TasksPerCta >= 2) {
    __syncthreads();
    NOWAG_GATE_UP_RUN_TASK_SLOT(1);
  }
  if constexpr (TasksPerCta >= 4) {
    __syncthreads();
    NOWAG_GATE_UP_RUN_TASK_SLOT(2);
    __syncthreads();
    NOWAG_GATE_UP_RUN_TASK_SLOT(3);
  }
  if constexpr (TasksPerCta >= 8) {
    __syncthreads();
    NOWAG_GATE_UP_RUN_TASK_SLOT(4);
    __syncthreads();
    NOWAG_GATE_UP_RUN_TASK_SLOT(5);
    __syncthreads();
    NOWAG_GATE_UP_RUN_TASK_SLOT(6);
    __syncthreads();
    NOWAG_GATE_UP_RUN_TASK_SLOT(7);
  }

#undef NOWAG_GATE_UP_RUN_TASK_SLOT
}

template <int NumWarps, bool PrecomputeTokens, int SharedEntries>
__global__ __launch_bounds__(NumWarps * kWarpSize)
void gate_up_exact_shared_codebook_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ gate_codebook,
    const int32_t* __restrict__ gate_assignments,
    const __nv_bfloat16* __restrict__ gate_input_norm,
    const __nv_bfloat16* __restrict__ gate_output_norm,
    __nv_bfloat16* __restrict__ gate_output,
    const __nv_bfloat16* __restrict__ up_codebook,
    const int32_t* __restrict__ up_assignments,
    const __nv_bfloat16* __restrict__ up_input_norm,
    const __nv_bfloat16* __restrict__ up_output_norm,
    __nv_bfloat16* __restrict__ up_output,
    const int32_t* __restrict__ sorted_tickets,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tickets_post_padded,
    int projection_count,
    int output_rows,
    int num_routes,
    int top_k,
    int in_features,
    int out_features,
    int output_stride,
    int output_start_lane,
    int num_groups,
    int num_words,
    int alignment_block_ratio) {
  constexpr int kBlockM = 16;
  constexpr int kMTiles = 1;
  constexpr int kBlockN = NumWarps * 16;
  constexpr int kCodebookSize = 1 << 12;
  constexpr int kABytes =
      2 * kBlockM * core::kPhysicalPacketK * sizeof(__nv_bfloat16);
  constexpr int kTokenBytes =
      PrecomputeTokens ? kBlockM * sizeof(int32_t) : 0;
  constexpr int kCodebookOffset = (kABytes + kTokenBytes + 15) & ~15;

  const int block_m_index = blockIdx.x;
  const int block_m_start = block_m_index * kBlockM;
  const int padded_rows = *num_tickets_post_padded;
  if (block_m_start >= padded_rows) {
    return;
  }
  const int tiles_per_projection = out_features / kBlockN;
  const int projection_index = blockIdx.y / tiles_per_projection;
  if (projection_index >= projection_count) {
    return;
  }
  const bool use_up = projection_index != 0;
  const auto* codebook = use_up ? up_codebook : gate_codebook;
  const auto* assignments = use_up ? up_assignments : gate_assignments;
  const auto* input_norm = use_up ? up_input_norm : gate_input_norm;
  const auto* output_norm = use_up ? up_output_norm : gate_output_norm;
  auto* projection_output = use_up ? up_output : gate_output;
  const int block_n_start =
      (blockIdx.y - projection_index * tiles_per_projection) * kBlockN;
  const int expert = expert_ids[block_m_index / alignment_block_ratio];
  const auto* expert_assignments = assignments +
      static_cast<int64_t>(expert) * num_words * out_features;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lane_group = lane >> 2;
  const int lane_in_group = lane & 3;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_raw);
  auto* shared_tokens = reinterpret_cast<int32_t*>(shared_raw + kABytes);
  auto* shared_codebook = reinterpret_cast<__nv_bfloat16*>(
      shared_raw + kCodebookOffset);

  // The prefix copy is issued first.  Its copy-engine work overlaps the
  // route table, accumulator initialization, and first normalized A packet.
  stage_codebook_prefix<SharedEntries>(shared_codebook, codebook);

  if constexpr (PrecomputeTokens) {
    for (int local_m = threadIdx.x; local_m < kBlockM;
         local_m += blockDim.x) {
      const int ticket = sorted_tickets[block_m_start + local_m];
      shared_tokens[local_m] =
          ticket >= 0 && ticket < num_routes ? ticket / top_k : -1;
    }
    __syncthreads();
  }

  float accumulators[kMTiles][2][4];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
#pragma unroll
    for (int value = 0; value < 4; ++value) {
      accumulators[0][n_half][value] = 0.0f;
    }
  }

  const int packets =
      (num_groups + core::kGroupsPerPacket - 1) /
      core::kGroupsPerPacket;
  const int full_packets = in_features / core::kExactPacketK;
  stage_gate_up_a_packet<kBlockM, NumWarps, PrecomputeTokens>(
      shared_a,
      shared_tokens,
      hidden,
      input_norm,
      sorted_tickets,
      expert,
      block_m_start,
      0,
      full_packets,
      num_routes,
      top_k,
      in_features,
      num_groups);
  if constexpr (SharedEntries > 0) {
    cp_async_wait_all();
  }
  __syncthreads();

  int read_pipe = 0;
  for (int packet = 0; packet < packets; ++packet) {
    const int next_packet = packet + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_packet < packets) {
      stage_gate_up_a_packet<kBlockM, NumWarps, PrecomputeTokens>(
          shared_a + write_pipe * kBlockM * core::kPhysicalPacketK,
          shared_tokens,
          hidden,
          input_norm,
          sorted_tickets,
          expert,
          block_m_start,
          next_packet,
          full_packets,
          num_routes,
          top_k,
          in_features,
          num_groups);
    }

    const auto* shared_stage =
        shared_a + read_pipe * kBlockM * core::kPhysicalPacketK;
    if (packet < full_packets) {
      compute_coalesced_k48_packet_cached<kBlockM, SharedEntries>(
          accumulators,
          shared_stage,
          codebook,
          shared_codebook,
          expert_assignments,
          packet,
          block_n_start,
          out_features,
          out_features,
          1);
    } else {
      compute_padded_tail_cached<kBlockM, SharedEntries>(
          accumulators,
          shared_stage,
          codebook,
          shared_codebook,
          expert_assignments,
          packet,
          block_n_start,
          out_features,
          kCodebookSize,
          num_groups,
          num_words,
          out_features,
          1);
    }
    if (next_packet < packets) {
      __syncthreads();
    }
    read_pipe = write_pipe;
  }

  float norm_low[2];
  float norm_high[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    norm_low[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n]);
    norm_high[n_half] = __bfloat162float(output_norm[
        static_cast<int64_t>(expert) * out_features + output_n + 1]);
  }

  const int row0 = block_m_start + lane_group;
  const int row1 = row0 + 8;
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const int output_n = block_n_start + warp * 16 + n_half * 8 +
        lane_in_group * 2;
    if (row0 < output_rows) {
      auto* destination = projection_output +
          static_cast<int64_t>(row0) * output_stride + output_start_lane +
          output_n;
      const auto values = __floats2bfloat162_rn(
          accumulators[0][n_half][0] * norm_low[n_half],
          accumulators[0][n_half][1] * norm_high[n_half]);
      if ((output_start_lane & 1) == 0) {
        *reinterpret_cast<__nv_bfloat162*>(destination) = values;
      } else {
        destination[0] = __low2bfloat16(values);
        destination[1] = __high2bfloat16(values);
      }
    }
    if (row1 < output_rows) {
      auto* destination = projection_output +
          static_cast<int64_t>(row1) * output_stride + output_start_lane +
          output_n;
      const auto values = __floats2bfloat162_rn(
          accumulators[0][n_half][2] * norm_low[n_half],
          accumulators[0][n_half][3] * norm_high[n_half]);
      if ((output_start_lane & 1) == 0) {
        *reinterpret_cast<__nv_bfloat162*>(destination) = values;
      } else {
        destination[0] = __low2bfloat16(values);
        destination[1] = __high2bfloat16(values);
      }
    }
  }
}

template <int BlockM>
__global__ void silu_down_norm_epilogue_kernel(
    __nv_bfloat16* gate_middle,
    const __nv_bfloat16* up_output,
    const __nv_bfloat16* down_input_norm,
    const int32_t* expert_ids,
    const int32_t* num_tickets_post_padded,
    int output_rows,
    int out_features,
    int output_stride,
    int output_start_lane,
    int alignment_block_ratio,
    float swiglu_limit,
    bool preapply_down_norm) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t total = static_cast<int64_t>(output_rows) * output_stride;
  if (linear >= total) {
    return;
  }
  const int row = linear / output_stride;
  const int physical_n =
      linear - static_cast<int64_t>(row) * output_stride;
  const int padded_rows = *num_tickets_post_padded;
  const int n = physical_n - output_start_lane;
  if (row >= padded_rows || n < 0 || n >= out_features) {
    gate_middle[linear] = from_float(0.0f);
    return;
  }
  const int expert = expert_ids[row / (BlockM * alignment_block_ratio)];
  float gate = __bfloat162float(gate_middle[linear]);
  float up = __bfloat162float(up_output[linear]);
  if (swiglu_limit > 0.0f) {
    gate = fminf(gate, swiglu_limit);
    up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
  }
  const float activated = gate / (1.0f + expf(-gate)) * up;
  const __nv_bfloat16 rounded = from_float(activated);
  if (preapply_down_norm) {
    const float scaled = __bfloat162float(rounded) * __bfloat162float(
        down_input_norm[static_cast<int64_t>(expert) * out_features + n]);
    gate_middle[linear] = from_float(scaled);
  } else {
    gate_middle[linear] = rounded;
  }
}

template <int BlockM, int NumWarps, bool PrecomputeTokens>
void launch_projection_batch(
    const torch::Tensor& hidden_states,
    const torch::Tensor& sorted_tickets,
    const torch::Tensor& expert_ids,
    const torch::Tensor& num_tickets_post_padded,
    const GateUpProjectionBatch& projection_batch,
    int output_rows,
    int num_routes,
    int top_k,
    int num_m_blocks,
    int alignment_block_ratio,
    int out_features,
    int output_stride,
    int output_start_lane,
    cudaStream_t stream) {
  const int in_features = hidden_states.size(1);
  const int num_groups =
      (in_features + core::kGroupSize - 1) / core::kGroupSize;
  const int num_words = (num_groups * 12 + 31) / 32;
  constexpr int kBlockN = NumWarps * 16;
  const dim3 grid(
      num_m_blocks, projection_batch.count * out_features / kBlockN);
  constexpr int kSharedBytes =
      2 * BlockM * core::kPhysicalPacketK * sizeof(__nv_bfloat16) +
      (PrecomputeTokens ? BlockM * sizeof(int32_t) : 0);
  gate_up_exact_projection_kernel<BlockM, NumWarps, PrecomputeTokens><<<
      grid,
      NumWarps * kWarpSize,
      kSharedBytes,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
      projection_batch.projections[0].codebook,
      projection_batch.projections[0].assignments,
      projection_batch.projections[0].input_norm,
      projection_batch.projections[0].output_norm,
      projection_batch.projections[0].output,
      projection_batch.projections[0].codebook_size,
      projection_batch.projections[0].codebook_banks,
      projection_batch.projections[1].codebook,
      projection_batch.projections[1].assignments,
      projection_batch.projections[1].input_norm,
      projection_batch.projections[1].output_norm,
      projection_batch.projections[1].output,
      projection_batch.projections[1].codebook_size,
      projection_batch.projections[1].codebook_banks,
      sorted_tickets.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      projection_batch.count,
      output_rows,
      num_routes,
      top_k,
      in_features,
      out_features,
      output_stride,
      output_start_lane,
      num_groups,
      num_words,
      alignment_block_ratio);
}

template <int BlockM, int NumWarps>
void launch_projection_batch_tasks(
    const torch::Tensor& hidden_states,
    const torch::Tensor& sorted_tickets,
    const torch::Tensor& expert_ids16,
    const torch::Tensor& num_tickets_post_padded,
    const torch::Tensor& tasks,
    const torch::Tensor& task_counts,
    int task_count_index,
    const GateUpProjectionBatch& projection_batch,
    int output_rows,
    int num_routes,
    int top_k,
    int out_features,
    int output_stride,
    int output_start_lane,
    int tasks_per_cta,
    cudaStream_t stream) {
  const int in_features = hidden_states.size(1);
  const int num_groups =
      (in_features + core::kGroupSize - 1) / core::kGroupSize;
  const int num_words = (num_groups * 12 + 31) / 32;
  constexpr int kBlockN = NumWarps * 16;
  const int task_slots = tasks.size(0);
  constexpr int kSharedBytes =
      2 * BlockM * core::kPhysicalPacketK * sizeof(__nv_bfloat16) +
      BlockM * sizeof(int32_t);
  if constexpr (BlockM == 16) {
    const auto launch_task_batch = [&](auto batch_size) {
      constexpr int kTasksPerCta = decltype(batch_size)::value;
      const int task_grid_slots = std::max(
          1, (task_slots + kTasksPerCta - 1) / kTasksPerCta);
      const dim3 grid(
          task_grid_slots, projection_batch.count * out_features / kBlockN);
      gate_up_exact_projection_bm16_task_batch_kernel<kTasksPerCta><<<
          grid,
          NumWarps * kWarpSize,
          kSharedBytes,
          stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
          projection_batch.projections[0].codebook,
          projection_batch.projections[0].assignments,
          projection_batch.projections[0].input_norm,
          projection_batch.projections[0].output_norm,
          projection_batch.projections[0].output,
          projection_batch.projections[0].codebook_size,
          projection_batch.projections[0].codebook_banks,
          projection_batch.projections[1].codebook,
          projection_batch.projections[1].assignments,
          projection_batch.projections[1].input_norm,
          projection_batch.projections[1].output_norm,
          projection_batch.projections[1].output,
          projection_batch.projections[1].codebook_size,
          projection_batch.projections[1].codebook_banks,
          sorted_tickets.data_ptr<int32_t>(),
          projection_batch.count,
          output_rows,
          num_routes,
          top_k,
          in_features,
          out_features,
          output_stride,
          output_start_lane,
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
        task_slots, projection_batch.count * out_features / kBlockN);
    gate_up_exact_projection_task_kernel<BlockM, NumWarps, true><<<
        grid,
        NumWarps * kWarpSize,
        kSharedBytes,
        stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        projection_batch.projections[0].codebook,
        projection_batch.projections[0].assignments,
        projection_batch.projections[0].input_norm,
        projection_batch.projections[0].output_norm,
        projection_batch.projections[0].output,
        projection_batch.projections[0].codebook_size,
        projection_batch.projections[0].codebook_banks,
        projection_batch.projections[1].codebook,
        projection_batch.projections[1].assignments,
        projection_batch.projections[1].input_norm,
        projection_batch.projections[1].output_norm,
        projection_batch.projections[1].output,
        projection_batch.projections[1].codebook_size,
        projection_batch.projections[1].codebook_banks,
        sorted_tickets.data_ptr<int32_t>(),
        projection_batch.count,
        output_rows,
        num_routes,
        top_k,
        in_features,
        out_features,
        output_stride,
        output_start_lane,
        num_groups,
        num_words,
        reinterpret_cast<const int2*>(tasks.data_ptr<int32_t>()),
        task_counts.data_ptr<int32_t>(),
        task_count_index);
  }
}

template <int NumWarps, bool PrecomputeTokens, int SharedEntries>
void launch_projection_batch_shared_codebook(
    const torch::Tensor& hidden_states,
    const torch::Tensor& sorted_tickets,
    const torch::Tensor& expert_ids,
    const torch::Tensor& num_tickets_post_padded,
    const GateUpProjectionBatch& projection_batch,
    int output_rows,
    int num_routes,
    int top_k,
    int num_m_blocks,
    int alignment_block_ratio,
    int out_features,
    int output_stride,
    int output_start_lane,
    cudaStream_t stream) {
  constexpr int kBlockM = 16;
  constexpr int kBlockN = NumWarps * 16;
  constexpr int kABytes =
      2 * kBlockM * core::kPhysicalPacketK * sizeof(__nv_bfloat16);
  constexpr int kTokenBytes =
      PrecomputeTokens ? kBlockM * sizeof(int32_t) : 0;
  constexpr int kCodebookOffset = (kABytes + kTokenBytes + 15) & ~15;
  constexpr int kSharedBytes = kCodebookOffset +
      SharedEntries * core::kGroupSize * sizeof(__nv_bfloat16);
  const int in_features = hidden_states.size(1);
  const int num_groups =
      (in_features + core::kGroupSize - 1) / core::kGroupSize;
  const int num_words = (num_groups * 12 + 31) / 32;
  const dim3 grid(
      num_m_blocks, projection_batch.count * out_features / kBlockN);

  if constexpr (kSharedBytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gate_up_exact_shared_codebook_kernel<
            NumWarps, PrecomputeTokens, SharedEntries>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
  }
  gate_up_exact_shared_codebook_kernel<
      NumWarps, PrecomputeTokens, SharedEntries><<<
      grid,
      NumWarps * kWarpSize,
      kSharedBytes,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
      projection_batch.projections[0].codebook,
      projection_batch.projections[0].assignments,
      projection_batch.projections[0].input_norm,
      projection_batch.projections[0].output_norm,
      projection_batch.projections[0].output,
      projection_batch.projections[1].codebook,
      projection_batch.projections[1].assignments,
      projection_batch.projections[1].input_norm,
      projection_batch.projections[1].output_norm,
      projection_batch.projections[1].output,
      sorted_tickets.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      projection_batch.count,
      output_rows,
      num_routes,
      top_k,
      in_features,
      out_features,
      output_stride,
      output_start_lane,
      num_groups,
      num_words,
      alignment_block_ratio);
}

template <int BlockM, int NumWarps, bool PrecomputeTokens>
void launch_gate_up(
    torch::Tensor hidden_states,
    torch::Tensor gate_codebook,
    torch::Tensor gate_packed_assignments,
    torch::Tensor gate_input_norm,
    torch::Tensor gate_output_norm,
    torch::Tensor up_codebook,
    torch::Tensor up_packed_assignments,
    torch::Tensor up_input_norm,
    torch::Tensor up_output_norm,
    torch::Tensor down_input_norm,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor gate_up_workspace,
    int num_routes,
    int top_k,
    int num_m_blocks,
    int alignment_block_ratio,
    int output_start_lane,
    bool fuse_projections,
    float swiglu_limit,
    bool preapply_down_norm,
    cudaStream_t stream) {
  const int out_features = gate_output_norm.size(1);
  const int output_stride = gate_up_workspace.size(1);
  const int output_rows = gate_up_workspace.size(0) / 2;
  auto* workspace = reinterpret_cast<__nv_bfloat16*>(
      gate_up_workspace.data_ptr());
  auto* gate_output = workspace;
  auto* up_output = workspace +
      static_cast<int64_t>(output_rows) * output_stride;

  GateUpProjectionBatch projection_batch{};
  projection_batch.projections[0] = GateUpProjectionDescriptor{
      reinterpret_cast<const __nv_bfloat16*>(gate_codebook.data_ptr()),
      gate_packed_assignments.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(gate_input_norm.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gate_output_norm.data_ptr()),
      gate_output,
      static_cast<int>(gate_codebook.size(1)),
      static_cast<int>(gate_codebook.size(0))};
  projection_batch.projections[1] = GateUpProjectionDescriptor{
      reinterpret_cast<const __nv_bfloat16*>(up_codebook.data_ptr()),
      up_packed_assignments.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(up_input_norm.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(up_output_norm.data_ptr()),
      up_output,
      static_cast<int>(up_codebook.size(1)),
      static_cast<int>(up_codebook.size(0))};

  if (fuse_projections) {
    projection_batch.count = 2;
    launch_projection_batch<BlockM, NumWarps, PrecomputeTokens>(
        hidden_states,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        projection_batch,
        output_rows,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        out_features,
        output_stride,
        output_start_lane,
        stream);
  } else {
    projection_batch.count = 1;
    launch_projection_batch<BlockM, NumWarps, PrecomputeTokens>(
        hidden_states,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        projection_batch,
        output_rows,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        out_features,
        output_stride,
        output_start_lane,
        stream);
    projection_batch.projections[0] = projection_batch.projections[1];
    launch_projection_batch<BlockM, NumWarps, PrecomputeTokens>(
        hidden_states,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        projection_batch,
        output_rows,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        out_features,
        output_stride,
        output_start_lane,
        stream);
  }

  constexpr int kThreads = 256;
  const int64_t epilogue_elements =
      static_cast<int64_t>(output_rows) * output_stride;
  const int epilogue_blocks = static_cast<int>(
      (epilogue_elements + kThreads - 1) / kThreads);
  silu_down_norm_epilogue_kernel<BlockM><<<
      epilogue_blocks,
      kThreads,
      0,
      stream>>>(
      gate_output,
      up_output,
      reinterpret_cast<const __nv_bfloat16*>(down_input_norm.data_ptr()),
      expert_ids.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      output_rows,
      out_features,
      output_stride,
      output_start_lane,
      alignment_block_ratio,
      swiglu_limit,
      preapply_down_norm);
}

void launch_gate_up_adaptive(
    torch::Tensor hidden_states,
    torch::Tensor gate_codebook,
    torch::Tensor gate_packed_assignments,
    torch::Tensor gate_input_norm,
    torch::Tensor gate_output_norm,
    torch::Tensor up_codebook,
    torch::Tensor up_packed_assignments,
    torch::Tensor up_input_norm,
    torch::Tensor up_output_norm,
    torch::Tensor down_input_norm,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor gate_up_workspace,
    torch::Tensor tasks64,
    torch::Tensor tasks16,
    torch::Tensor task_counts,
    int num_routes,
    int top_k,
    int output_start_lane,
    int tasks_per_cta,
    bool fuse_projections,
    float swiglu_limit,
    bool preapply_down_norm,
    cudaStream_t stream) {
  const int out_features = gate_output_norm.size(1);
  const int output_stride = gate_up_workspace.size(1);
  const int output_rows = gate_up_workspace.size(0) / 2;
  auto* workspace = reinterpret_cast<__nv_bfloat16*>(
      gate_up_workspace.data_ptr());
  auto* gate_output = workspace;
  auto* up_output = workspace +
      static_cast<int64_t>(output_rows) * output_stride;

  GateUpProjectionBatch projection_batch{};
  projection_batch.projections[0] = GateUpProjectionDescriptor{
      reinterpret_cast<const __nv_bfloat16*>(gate_codebook.data_ptr()),
      gate_packed_assignments.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(gate_input_norm.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gate_output_norm.data_ptr()),
      gate_output,
      static_cast<int>(gate_codebook.size(1)),
      static_cast<int>(gate_codebook.size(0))};
  projection_batch.projections[1] = GateUpProjectionDescriptor{
      reinterpret_cast<const __nv_bfloat16*>(up_codebook.data_ptr()),
      up_packed_assignments.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(up_input_norm.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(up_output_norm.data_ptr()),
      up_output,
      static_cast<int>(up_codebook.size(1)),
      static_cast<int>(up_codebook.size(0))};

  const auto launch_queues = [&](const GateUpProjectionBatch& batch) {
    launch_projection_batch_tasks<64, 8>(
        hidden_states,
        sorted_tickets,
        expert_ids16,
        num_tickets_post_padded,
        tasks64,
        task_counts,
        0,
        batch,
        output_rows,
        num_routes,
        top_k,
        out_features,
        output_stride,
        output_start_lane,
        1,
        stream);
    launch_projection_batch_tasks<16, 8>(
        hidden_states,
        sorted_tickets,
        expert_ids16,
        num_tickets_post_padded,
        tasks16,
        task_counts,
        1,
        batch,
        output_rows,
        num_routes,
        top_k,
        out_features,
        output_stride,
        output_start_lane,
        tasks_per_cta,
        stream);
  };
  if (fuse_projections) {
    projection_batch.count = 2;
    launch_queues(projection_batch);
  } else {
    projection_batch.count = 1;
    launch_queues(projection_batch);
    projection_batch.projections[0] = projection_batch.projections[1];
    launch_queues(projection_batch);
  }

  constexpr int kThreads = 256;
  const int64_t epilogue_elements =
      static_cast<int64_t>(output_rows) * output_stride;
  const int epilogue_blocks = static_cast<int>(
      (epilogue_elements + kThreads - 1) / kThreads);
  silu_down_norm_epilogue_kernel<16><<<
      epilogue_blocks,
      kThreads,
      0,
      stream>>>(
      gate_output,
      up_output,
      reinterpret_cast<const __nv_bfloat16*>(down_input_norm.data_ptr()),
      expert_ids16.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      output_rows,
      out_features,
      output_stride,
      output_start_lane,
      1,
      swiglu_limit,
      preapply_down_norm);
}

template <int NumWarps, bool PrecomputeTokens, int SharedEntries>
void launch_gate_up_shared_codebook(
    torch::Tensor hidden_states,
    torch::Tensor gate_codebook,
    torch::Tensor gate_packed_assignments,
    torch::Tensor gate_input_norm,
    torch::Tensor gate_output_norm,
    torch::Tensor up_codebook,
    torch::Tensor up_packed_assignments,
    torch::Tensor up_input_norm,
    torch::Tensor up_output_norm,
    torch::Tensor down_input_norm,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor gate_up_workspace,
    int num_routes,
    int top_k,
    int num_m_blocks,
    int alignment_block_ratio,
    int output_start_lane,
    bool fuse_projections,
    float swiglu_limit,
    bool preapply_down_norm,
    cudaStream_t stream) {
  const int out_features = gate_output_norm.size(1);
  const int output_stride = gate_up_workspace.size(1);
  const int output_rows = gate_up_workspace.size(0) / 2;
  auto* workspace = reinterpret_cast<__nv_bfloat16*>(
      gate_up_workspace.data_ptr());
  auto* gate_output = workspace;
  auto* up_output = workspace +
      static_cast<int64_t>(output_rows) * output_stride;

  GateUpProjectionBatch projection_batch{};
  projection_batch.projections[0] = GateUpProjectionDescriptor{
      reinterpret_cast<const __nv_bfloat16*>(gate_codebook.data_ptr()),
      gate_packed_assignments.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(gate_input_norm.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(gate_output_norm.data_ptr()),
      gate_output,
      static_cast<int>(gate_codebook.size(1)),
      static_cast<int>(gate_codebook.size(0))};
  projection_batch.projections[1] = GateUpProjectionDescriptor{
      reinterpret_cast<const __nv_bfloat16*>(up_codebook.data_ptr()),
      up_packed_assignments.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(up_input_norm.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(up_output_norm.data_ptr()),
      up_output,
      static_cast<int>(up_codebook.size(1)),
      static_cast<int>(up_codebook.size(0))};

  if (fuse_projections) {
    projection_batch.count = 2;
    launch_projection_batch_shared_codebook<
        NumWarps, PrecomputeTokens, SharedEntries>(
        hidden_states,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        projection_batch,
        output_rows,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        out_features,
        output_stride,
        output_start_lane,
        stream);
  } else {
    projection_batch.count = 1;
    launch_projection_batch_shared_codebook<
        NumWarps, PrecomputeTokens, SharedEntries>(
        hidden_states,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        projection_batch,
        output_rows,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        out_features,
        output_stride,
        output_start_lane,
        stream);
    projection_batch.projections[0] = projection_batch.projections[1];
    launch_projection_batch_shared_codebook<
        NumWarps, PrecomputeTokens, SharedEntries>(
        hidden_states,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        projection_batch,
        output_rows,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        out_features,
        output_stride,
        output_start_lane,
        stream);
  }

  constexpr int kThreads = 256;
  const int64_t epilogue_elements =
      static_cast<int64_t>(output_rows) * output_stride;
  const int epilogue_blocks = static_cast<int>(
      (epilogue_elements + kThreads - 1) / kThreads);
  silu_down_norm_epilogue_kernel<16><<<
      epilogue_blocks,
      kThreads,
      0,
      stream>>>(
      gate_output,
      up_output,
      reinterpret_cast<const __nv_bfloat16*>(down_input_norm.data_ptr()),
      expert_ids.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      output_rows,
      out_features,
      output_stride,
      output_start_lane,
      alignment_block_ratio,
      swiglu_limit,
      preapply_down_norm);
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

void nowag_moe_gate_up_exact_k48_adaptive_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_codebook,
    torch::Tensor gate_packed_assignments,
    torch::Tensor gate_input_norm,
    torch::Tensor gate_output_norm,
    torch::Tensor up_codebook,
    torch::Tensor up_packed_assignments,
    torch::Tensor up_input_norm,
    torch::Tensor up_output_norm,
    torch::Tensor down_input_norm,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor gate_up_workspace,
    torch::Tensor tasks64,
    torch::Tensor tasks16,
    torch::Tensor task_counts,
    int64_t num_routes,
    int64_t top_k,
    int64_t block_n,
    int64_t tasks_per_cta,
    int64_t output_start_lane,
    bool fuse_projections,
    double swiglu_limit,
    bool preapply_down_norm) {
  TORCH_CHECK(block_n == 128,
              "adaptive Gate/Up V1 requires block_n=128");
  TORCH_CHECK(
      tasks_per_cta == 1 || tasks_per_cta == 2 || tasks_per_cta == 4 ||
          tasks_per_cta == 8,
      "tasks_per_cta must be 1, 2, 4, or 8");
  TORCH_CHECK(num_routes > 0 && top_k > 0,
              "route dimensions must be positive");
  TORCH_CHECK(output_start_lane >= 0,
              "output_start_lane must be non-negative");
  TORCH_CHECK(swiglu_limit >= 0.0,
              "swiglu_limit must be zero (disabled) or positive");

  check_tensor(hidden_states, "hidden_states", at::kBFloat16);
  check_tensor(gate_codebook, "gate_codebook", at::kBFloat16);
  check_tensor(
      gate_packed_assignments, "gate_packed_assignments", at::kInt);
  check_tensor(gate_input_norm, "gate_input_norm", at::kBFloat16);
  check_tensor(gate_output_norm, "gate_output_norm", at::kBFloat16);
  check_tensor(up_codebook, "up_codebook", at::kBFloat16);
  check_tensor(up_packed_assignments, "up_packed_assignments", at::kInt);
  check_tensor(up_input_norm, "up_input_norm", at::kBFloat16);
  check_tensor(up_output_norm, "up_output_norm", at::kBFloat16);
  check_tensor(down_input_norm, "down_input_norm", at::kBFloat16);
  check_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_tensor(expert_ids16, "expert_ids16", at::kInt);
  check_tensor(
      num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  check_tensor(gate_up_workspace, "gate_up_workspace", at::kBFloat16);
  check_tensor(tasks64, "tasks64", at::kInt);
  check_tensor(tasks16, "tasks16", at::kInt);
  check_tensor(task_counts, "task_counts", at::kInt);

  TORCH_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) > 0 &&
                  hidden_states.size(1) % 2 == 0,
              "hidden_states must be [M,K] with positive even K");
  TORCH_CHECK(gate_codebook.dim() == 3 &&
                  gate_codebook.size(2) == core::kGroupSize,
              "gate_codebook must be [B,C,6]");
  TORCH_CHECK(up_codebook.dim() == 3 &&
                  up_codebook.size(2) == core::kGroupSize,
              "up_codebook must be [B,C,6]");
  TORCH_CHECK(gate_output_norm.dim() == 2 &&
                  up_output_norm.sizes() == gate_output_norm.sizes(),
              "Gate and Up output norms must have the same [E,N] shape");
  const int64_t num_experts = gate_output_norm.size(0);
  const int64_t out_features = gate_output_norm.size(1);
  TORCH_CHECK(num_experts > 0 && out_features > 0 &&
                  out_features % block_n == 0,
              "adaptive Gate/Up requires positive E and N divisible by 128");
  const auto check_codebook = [&](const torch::Tensor& codebook,
                                  const char* name) {
    TORCH_CHECK(codebook.size(0) == 1 || codebook.size(0) == num_experts,
                name, " bank dimension must be 1 or match private expert count ",
                num_experts);
    TORCH_CHECK(codebook.size(1) > 0 && codebook.size(1) <= (1 << 12),
                name, " size must fit 12-bit assignments");
  };
  check_codebook(gate_codebook, "gate_codebook");
  check_codebook(up_codebook, "up_codebook");
  TORCH_CHECK(gate_input_norm.sizes() ==
                  torch::IntArrayRef({num_experts, hidden_states.size(1)}),
              "gate_input_norm must be private [E,K]");
  TORCH_CHECK(up_input_norm.sizes() == gate_input_norm.sizes(),
              "up_input_norm must be private [E,K]");
  TORCH_CHECK(down_input_norm.sizes() == gate_output_norm.sizes(),
              "down_input_norm must be private [E,N]");
  TORCH_CHECK(gate_up_workspace.dim() == 2 &&
                  gate_up_workspace.size(0) % 2 == 0 &&
                  gate_up_workspace.size(1) % 2 == 0 &&
                  gate_up_workspace.size(1) >=
                      output_start_lane + out_features,
              "gate_up_workspace must be [2*rows,physical_N] with even "
              "physical_N >= output_start_lane + logical_N");
  const int64_t output_rows = gate_up_workspace.size(0) / 2;
  TORCH_CHECK(output_rows >= 16,
              "adaptive Gate/Up workspace must hold at least one BM16 tile");
  TORCH_CHECK(sorted_tickets.dim() == 1 &&
                  sorted_tickets.numel() >= output_rows,
              "sorted_tickets must cover the Gate/Up workspace rows");
  TORCH_CHECK(expert_ids16.dim() == 1 &&
                  expert_ids16.numel() >= output_rows / 16,
              "expert_ids16 must cover every BM16 workspace block");
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

  const int64_t num_groups =
      (hidden_states.size(1) + core::kGroupSize - 1) / core::kGroupSize;
  const int64_t expected_words = (num_groups * 12 + 31) / 32;
  const auto check_assignments = [&](const torch::Tensor& assignments,
                                     const char* name) {
    TORCH_CHECK(assignments.dim() == 3 &&
                    assignments.size(0) == num_experts &&
                    assignments.size(1) == expected_words &&
                    assignments.size(2) == out_features,
                name, " must retain the private word-major [E,W,N] layout");
  };
  check_assignments(gate_packed_assignments, "gate_packed_assignments");
  check_assignments(up_packed_assignments, "up_packed_assignments");

  const auto device = hidden_states.device();
  const torch::Tensor tensors[] = {
      gate_codebook, gate_packed_assignments, gate_input_norm,
      gate_output_norm, up_codebook, up_packed_assignments, up_input_norm,
      up_output_norm, down_input_norm, sorted_tickets, expert_ids16,
      num_tickets_post_padded, gate_up_workspace, tasks64, tasks16,
      task_counts};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device,
                "all adaptive Gate/Up tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_gate_up_adaptive(
      hidden_states,
      gate_codebook,
      gate_packed_assignments,
      gate_input_norm,
      gate_output_norm,
      up_codebook,
      up_packed_assignments,
      up_input_norm,
      up_output_norm,
      down_input_norm,
      sorted_tickets,
      expert_ids16,
      num_tickets_post_padded,
      gate_up_workspace,
      tasks64,
      tasks16,
      task_counts,
      static_cast<int>(num_routes),
      static_cast<int>(top_k),
      static_cast<int>(output_start_lane),
      static_cast<int>(tasks_per_cta),
      fuse_projections,
      static_cast<float>(swiglu_limit),
      preapply_down_norm,
      stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nowag_moe_gate_up_exact_k48_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_codebook,
    torch::Tensor gate_packed_assignments,
    torch::Tensor gate_input_norm,
    torch::Tensor gate_output_norm,
    torch::Tensor up_codebook,
    torch::Tensor up_packed_assignments,
    torch::Tensor up_input_norm,
    torch::Tensor up_output_norm,
    torch::Tensor down_input_norm,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor gate_up_workspace,
    int64_t num_routes,
    int64_t top_k,
    int64_t num_m_blocks,
    int64_t alignment_block_ratio,
    int64_t block_m,
    int64_t block_n,
    bool precompute_tokens,
    int64_t output_start_lane,
    bool fuse_projections,
    double swiglu_limit,
    bool preapply_down_norm) {
  // These are compiled candidate specializations, not runtime policy.  The
  // host selector chooses among offline-measured plans and falls back safely
  // when no exact GPU/shape/M profile matches.
  TORCH_CHECK(
      block_m == 16 || block_m == 32 || block_m == 64 || block_m == 128,
      "block_m must be 16, 32, 64, or 128");
  TORCH_CHECK(block_n == 64 || block_n == 128,
              "block_n must be 64 or 128");
  TORCH_CHECK(num_routes > 0 && top_k > 0 && num_m_blocks > 0,
              "route and launch dimensions must be positive");
  TORCH_CHECK(alignment_block_ratio > 0,
              "alignment_block_ratio must be positive");
  TORCH_CHECK(swiglu_limit >= 0.0,
              "swiglu_limit must be zero (disabled) or positive");

  check_tensor(hidden_states, "hidden_states", at::kBFloat16);
  check_tensor(gate_codebook, "gate_codebook", at::kBFloat16);
  check_tensor(gate_packed_assignments, "gate_packed_assignments", at::kInt);
  check_tensor(gate_input_norm, "gate_input_norm", at::kBFloat16);
  check_tensor(gate_output_norm, "gate_output_norm", at::kBFloat16);
  check_tensor(up_codebook, "up_codebook", at::kBFloat16);
  check_tensor(up_packed_assignments, "up_packed_assignments", at::kInt);
  check_tensor(up_input_norm, "up_input_norm", at::kBFloat16);
  check_tensor(up_output_norm, "up_output_norm", at::kBFloat16);
  check_tensor(down_input_norm, "down_input_norm", at::kBFloat16);
  check_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_tensor(expert_ids, "expert_ids", at::kInt);
  check_tensor(num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  check_tensor(gate_up_workspace, "gate_up_workspace", at::kBFloat16);

  TORCH_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) > 0 &&
                  hidden_states.size(1) % 2 == 0,
              "hidden_states must be [M,K] with positive even K");
  TORCH_CHECK(gate_codebook.dim() == 3 &&
                  gate_codebook.size(2) == core::kGroupSize,
              "gate_codebook must be [B,C,6]");
  TORCH_CHECK(up_codebook.dim() == 3 &&
                  up_codebook.size(2) == core::kGroupSize,
              "up_codebook must be [B,C,6]");
  TORCH_CHECK(gate_output_norm.dim() == 2 &&
                  up_output_norm.sizes() == gate_output_norm.sizes(),
              "Gate and Up output norms must have the same [E,N] shape");
  const int64_t num_experts = gate_output_norm.size(0);
  const int64_t out_features = gate_output_norm.size(1);
  TORCH_CHECK(num_experts > 0,
              "Gate/Up private tensors must contain at least one expert");
  const auto check_codebook = [&](const torch::Tensor& codebook,
                                  const char* name) {
    TORCH_CHECK(codebook.size(0) == 1 || codebook.size(0) == num_experts,
                name, " bank dimension must be 1 or match private expert count ",
                num_experts);
    TORCH_CHECK(codebook.size(1) > 0 && codebook.size(1) <= (1 << 12),
                name, " size must fit 12-bit assignments");
  };
  check_codebook(gate_codebook, "gate_codebook");
  check_codebook(up_codebook, "up_codebook");
  TORCH_CHECK(output_start_lane >= 0,
              "output_start_lane must be non-negative");
  TORCH_CHECK(out_features > 0 && out_features % block_n == 0,
              "Gate/Up N must be divisible by block_n");
  TORCH_CHECK(gate_input_norm.sizes() ==
                  torch::IntArrayRef({num_experts,
                                      hidden_states.size(1)}),
              "gate_input_norm must be private [E,K]");
  TORCH_CHECK(up_input_norm.sizes() == gate_input_norm.sizes(),
              "up_input_norm must be private [E,K]");
  TORCH_CHECK(down_input_norm.sizes() == gate_output_norm.sizes(),
              "down_input_norm must be private [E,N]");
  TORCH_CHECK(gate_up_workspace.dim() == 2 &&
                  gate_up_workspace.size(0) % 2 == 0 &&
                  gate_up_workspace.size(1) % 2 == 0 &&
                  gate_up_workspace.size(1) >=
                      output_start_lane + out_features,
              "gate_up_workspace must be [2*rows,physical_N] with even "
              "physical_N >= output_start_lane + logical_N");
  const int64_t output_rows = gate_up_workspace.size(0) / 2;
  TORCH_CHECK(sorted_tickets.numel() >= output_rows,
              "sorted_tickets is too small for the workspace");
  TORCH_CHECK(expert_ids.numel() * alignment_block_ratio >= num_m_blocks,
              "expert_ids is too small for the Gate/Up M blocks");
  TORCH_CHECK(num_tickets_post_padded.numel() == 1,
              "num_tickets_post_padded must be scalar");

  const int64_t num_groups =
      (hidden_states.size(1) + core::kGroupSize - 1) / core::kGroupSize;
  const int64_t expected_words = (num_groups * 12 + 31) / 32;
  const auto check_assignments = [&](const torch::Tensor& assignments,
                                     const char* name) {
    TORCH_CHECK(assignments.dim() == 3 &&
                    assignments.size(0) == num_experts &&
                    assignments.size(1) == expected_words &&
                    assignments.size(2) == out_features,
                name, " must retain the private word-major [E,W,N] layout");
  };
  check_assignments(gate_packed_assignments, "gate_packed_assignments");
  check_assignments(up_packed_assignments, "up_packed_assignments");

  const auto device = hidden_states.device();
  const torch::Tensor tensors[] = {
      gate_codebook, gate_packed_assignments, gate_input_norm,
      gate_output_norm, up_codebook, up_packed_assignments, up_input_norm,
      up_output_norm, down_input_norm, sorted_tickets, expert_ids,
      num_tickets_post_padded, gate_up_workspace};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device,
                "all Gate/Up tensors must be on the same CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto launch_variant = [&](auto precompute_tag) {
    constexpr bool kPrecomputeTokens = decltype(precompute_tag)::value;
    if (block_m == 16 && block_n == 64) {
      launch_gate_up<16, 4, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    } else if (block_m == 16 && block_n == 128) {
      launch_gate_up<16, 8, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    } else if (block_m == 32 && block_n == 64) {
      launch_gate_up<32, 4, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    } else if (block_m == 32 && block_n == 128) {
      launch_gate_up<32, 8, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    } else if (block_m == 64 && block_n == 64) {
      launch_gate_up<64, 4, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    } else if (block_m == 64 && block_n == 128) {
      launch_gate_up<64, 8, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    } else if (block_n == 64) {
      launch_gate_up<128, 4, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    } else {
      launch_gate_up<128, 8, kPrecomputeTokens>(
          hidden_states, gate_codebook, gate_packed_assignments,
          gate_input_norm, gate_output_norm, up_codebook,
          up_packed_assignments, up_input_norm, up_output_norm, down_input_norm,
          sorted_tickets, expert_ids, num_tickets_post_padded,
          gate_up_workspace, num_routes, top_k, num_m_blocks,
          alignment_block_ratio, output_start_lane, fuse_projections,
          static_cast<float>(swiglu_limit), preapply_down_norm, stream);
    }
  };
  if (precompute_tokens) {
    launch_variant(std::true_type{});
  } else {
    launch_variant(std::false_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nowag_moe_gate_up_exact_k48_codebook_cache_cuda(
    torch::Tensor hidden_states,
    torch::Tensor gate_codebook,
    torch::Tensor gate_packed_assignments,
    torch::Tensor gate_input_norm,
    torch::Tensor gate_output_norm,
    torch::Tensor up_codebook,
    torch::Tensor up_packed_assignments,
    torch::Tensor up_input_norm,
    torch::Tensor up_output_norm,
    torch::Tensor down_input_norm,
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor gate_up_workspace,
    int64_t num_routes,
    int64_t top_k,
    int64_t num_m_blocks,
    int64_t alignment_block_ratio,
    int64_t block_m,
    int64_t block_n,
    bool precompute_tokens,
    int64_t output_start_lane,
    bool fuse_projections,
    double swiglu_limit,
    bool preapply_down_norm,
    int64_t shared_codebook_entries,
    bool assignment_l2_only) {
  TORCH_CHECK(
      shared_codebook_entries == 0 || shared_codebook_entries == 4096,
      "shared_codebook_entries must be 0 or 4096");
  TORCH_CHECK(
      shared_codebook_entries == 0 || assignment_l2_only,
      "shared codebook candidates require assignment_l2_only=true");
  if (shared_codebook_entries == 0 && !assignment_l2_only) {
    nowag_moe_gate_up_exact_k48_cuda(
        hidden_states,
        gate_codebook,
        gate_packed_assignments,
        gate_input_norm,
        gate_output_norm,
        up_codebook,
        up_packed_assignments,
        up_input_norm,
        up_output_norm,
        down_input_norm,
        sorted_tickets,
        expert_ids,
        num_tickets_post_padded,
        gate_up_workspace,
        num_routes,
        top_k,
        num_m_blocks,
        alignment_block_ratio,
        block_m,
        block_n,
        precompute_tokens,
        output_start_lane,
        fuse_projections,
        swiglu_limit,
        preapply_down_norm);
    return;
  }

  TORCH_CHECK(block_m == 16,
              "shared codebook candidates require block_m=16");
  TORCH_CHECK(block_n == 64 || block_n == 128,
              "block_n must be 64 or 128");
  TORCH_CHECK(num_routes > 0 && top_k > 0 && num_m_blocks > 0,
              "route and launch dimensions must be positive");
  TORCH_CHECK(alignment_block_ratio > 0,
              "alignment_block_ratio must be positive");
  TORCH_CHECK(swiglu_limit >= 0.0,
              "swiglu_limit must be zero (disabled) or positive");

  check_tensor(hidden_states, "hidden_states", at::kBFloat16);
  check_tensor(gate_codebook, "gate_codebook", at::kBFloat16);
  check_tensor(
      gate_packed_assignments, "gate_packed_assignments", at::kInt);
  check_tensor(gate_input_norm, "gate_input_norm", at::kBFloat16);
  check_tensor(gate_output_norm, "gate_output_norm", at::kBFloat16);
  check_tensor(up_codebook, "up_codebook", at::kBFloat16);
  check_tensor(up_packed_assignments, "up_packed_assignments", at::kInt);
  check_tensor(up_input_norm, "up_input_norm", at::kBFloat16);
  check_tensor(up_output_norm, "up_output_norm", at::kBFloat16);
  check_tensor(down_input_norm, "down_input_norm", at::kBFloat16);
  check_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_tensor(expert_ids, "expert_ids", at::kInt);
  check_tensor(
      num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  check_tensor(gate_up_workspace, "gate_up_workspace", at::kBFloat16);

  TORCH_CHECK(hidden_states.dim() == 2 && hidden_states.size(1) > 0 &&
                  hidden_states.size(1) % 2 == 0,
              "hidden_states must be [M,K] with positive even K");
  TORCH_CHECK(
      gate_codebook.sizes() == torch::IntArrayRef({1, 4096, 6}),
      "gate_codebook must be the shared [1,4096,6] BF16 codebook");
  TORCH_CHECK(
      up_codebook.sizes() == torch::IntArrayRef({1, 4096, 6}),
      "up_codebook must be the shared [1,4096,6] BF16 codebook");
  TORCH_CHECK(gate_output_norm.dim() == 2 &&
                  up_output_norm.sizes() == gate_output_norm.sizes(),
              "Gate and Up output norms must have the same [E,N] shape");
  const int64_t num_experts = gate_output_norm.size(0);
  const int64_t out_features = gate_output_norm.size(1);
  TORCH_CHECK(num_experts > 0,
              "Gate/Up private tensors must contain at least one expert");
  TORCH_CHECK(output_start_lane >= 0,
              "output_start_lane must be non-negative");
  TORCH_CHECK(out_features > 0 && out_features % block_n == 0,
              "Gate/Up N must be divisible by block_n");
  TORCH_CHECK(gate_input_norm.sizes() ==
                  torch::IntArrayRef(
                      {num_experts, hidden_states.size(1)}),
              "gate_input_norm must be private [E,K]");
  TORCH_CHECK(up_input_norm.sizes() == gate_input_norm.sizes(),
              "up_input_norm must be private [E,K]");
  TORCH_CHECK(down_input_norm.sizes() == gate_output_norm.sizes(),
              "down_input_norm must be private [E,N]");
  TORCH_CHECK(gate_up_workspace.dim() == 2 &&
                  gate_up_workspace.size(0) % 2 == 0 &&
                  gate_up_workspace.size(1) % 2 == 0 &&
                  gate_up_workspace.size(1) >=
                      output_start_lane + out_features,
              "gate_up_workspace must be [2*rows,physical_N] with even "
              "physical_N >= output_start_lane + logical_N");
  const int64_t output_rows = gate_up_workspace.size(0) / 2;
  TORCH_CHECK(sorted_tickets.numel() >= output_rows,
              "sorted_tickets is too small for the workspace");
  TORCH_CHECK(expert_ids.numel() * alignment_block_ratio >= num_m_blocks,
              "expert_ids is too small for the Gate/Up M blocks");
  TORCH_CHECK(num_tickets_post_padded.numel() == 1,
              "num_tickets_post_padded must be scalar");

  const int64_t num_groups =
      (hidden_states.size(1) + core::kGroupSize - 1) /
      core::kGroupSize;
  const int64_t expected_words = (num_groups * 12 + 31) / 32;
  const auto check_assignments = [&](const torch::Tensor& assignments,
                                     const char* name) {
    TORCH_CHECK(assignments.dim() == 3 &&
                    assignments.size(0) == num_experts &&
                    assignments.size(1) == expected_words &&
                    assignments.size(2) == out_features,
                name,
                " must retain the private word-major [E,W,N] layout");
  };
  check_assignments(
      gate_packed_assignments, "gate_packed_assignments");
  check_assignments(up_packed_assignments, "up_packed_assignments");

  const auto device = hidden_states.device();
  const torch::Tensor tensors[] = {
      gate_codebook,
      gate_packed_assignments,
      gate_input_norm,
      gate_output_norm,
      up_codebook,
      up_packed_assignments,
      up_input_norm,
      up_output_norm,
      down_input_norm,
      sorted_tickets,
      expert_ids,
      num_tickets_post_padded,
      gate_up_workspace};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == device,
                "all Gate/Up tensors must be on the same CUDA device");
  }

  c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto launch_variant = [&](auto entries_tag, auto precompute_tag) {
    constexpr int kSharedEntries = decltype(entries_tag)::value;
    constexpr bool kPrecomputeTokens = decltype(precompute_tag)::value;
    if (block_n == 64) {
      launch_gate_up_shared_codebook<
          4, kPrecomputeTokens, kSharedEntries>(
          hidden_states,
          gate_codebook,
          gate_packed_assignments,
          gate_input_norm,
          gate_output_norm,
          up_codebook,
          up_packed_assignments,
          up_input_norm,
          up_output_norm,
          down_input_norm,
          sorted_tickets,
          expert_ids,
          num_tickets_post_padded,
          gate_up_workspace,
          num_routes,
          top_k,
          num_m_blocks,
          alignment_block_ratio,
          output_start_lane,
          fuse_projections,
          static_cast<float>(swiglu_limit),
          preapply_down_norm,
          stream);
    } else {
      launch_gate_up_shared_codebook<
          8, kPrecomputeTokens, kSharedEntries>(
          hidden_states,
          gate_codebook,
          gate_packed_assignments,
          gate_input_norm,
          gate_output_norm,
          up_codebook,
          up_packed_assignments,
          up_input_norm,
          up_output_norm,
          down_input_norm,
          sorted_tickets,
          expert_ids,
          num_tickets_post_padded,
          gate_up_workspace,
          num_routes,
          top_k,
          num_m_blocks,
          alignment_block_ratio,
          output_start_lane,
          fuse_projections,
          static_cast<float>(swiglu_limit),
          preapply_down_norm,
          stream);
    }
  };
  auto dispatch_entries = [&](auto precompute_tag) {
    if (shared_codebook_entries == 0) {
      launch_variant(std::integral_constant<int, 0>{}, precompute_tag);
    } else {
      launch_variant(
          std::integral_constant<int, 4096>{}, precompute_tag);
    }
  };
  if (precompute_tokens) {
    dispatch_entries(std::true_type{});
  } else {
    dispatch_entries(std::false_type{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
