#pragma once

#include <cuda_bf16.h>

#include <cstdint>

// Projection-agnostic NoWag D6/12-bit lookup + Tensor Core primitives.
//
// The caller owns A-tile staging and the projection-specific epilogue.  This
// core consumes one shared-memory A packet, reads one word-major assignment
// packet (3 uint32 -> 8 IDs), looks up eight D6 codewords, assembles three
// K16 operands, and accumulates three BF16 Tensor Core MMAs.
namespace nowag::k48_core {

constexpr int kGroupSize = 6;
constexpr int kPaddedGroupSize = 8;
constexpr int kGroupsPerPacket = 8;
constexpr int kExactPacketK = kGroupsPerPacket * kGroupSize;
constexpr int kPhysicalPacketK = kGroupsPerPacket * kPaddedGroupSize;

__device__ __forceinline__ uint32_t load_u32(
    const __nv_bfloat16* pointer) {
  return *reinterpret_cast<const uint32_t*>(pointer);
}

__device__ __forceinline__ uint32_t load_assignment_word(
    const int32_t* assignments,
    int word,
    int output_n,
    int assignment_word_stride,
    int assignment_output_stride) {
  return static_cast<uint32_t>(assignments[
      static_cast<int64_t>(word) * assignment_word_stride +
      static_cast<int64_t>(output_n) * assignment_output_stride]);
}

__device__ __forceinline__ int swizzled_a_offset(int row, int column) {
  // CUTLASS/CuTe's Ampere small-K layout:
  //   Swizzle<2,3,3> o Layout<Shape<8,32>, Stride<32,1>>
  // K64 is two independently swizzled K32 atoms. Keeping the explicit
  // formula here makes both the producer and ldmatrix consumer share one
  // auditable physical layout.
  const int row_atom = row >> 3;
  const int row_in_atom = row & 7;
  const int k_atom = column >> 5;
  const int k_in_atom = column & 31;
  const int logical = row_in_atom * 32 + k_in_atom;
  const int swizzled = logical ^ ((logical & 0xC0) >> 3);
  return (row_atom * 2 + k_atom) * (8 * 32) + swizzled;
}

__device__ __forceinline__ void load_a_fragment(
    uint32_t (&a)[4],
    const __nv_bfloat16* stage,
    int m_tile,
    int k_fragment,
    int lane) {
  // ldmatrix.x4 consumes four 8x8 quadrants. The first 16 lanes provide
  // addresses for K[0:8], and the second 16 provide K[8:16].
  const int row = m_tile * 16 + (lane & 15);
  const int column = k_fragment * 16 + (lane >> 4) * 8;
  const uint32_t shared_address = static_cast<uint32_t>(
      __cvta_generic_to_shared(stage + swizzled_a_offset(row, column)));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(shared_address));
#endif
}

__device__ __forceinline__ uint32_t decode_packet_id(
    uint32_t word0,
    uint32_t word1,
    uint32_t word2,
    int lane) {
  switch (lane) {
    case 0: return word0 & 0xFFFu;
    case 1: return (word0 >> 12) & 0xFFFu;
    case 2: return ((word0 >> 24) | (word1 << 8)) & 0xFFFu;
    case 3: return (word1 >> 4) & 0xFFFu;
    case 4: return (word1 >> 16) & 0xFFFu;
    case 5: return ((word1 >> 28) | (word2 << 4)) & 0xFFFu;
    case 6: return (word2 >> 8) & 0xFFFu;
    default: return (word2 >> 20) & 0xFFFu;
  }
}

template <int StartPair>
__device__ __forceinline__ uint32_t load_exact_b_operand(
    uint32_t word0,
    uint32_t word1,
    uint32_t word2,
    int lane_in_group,
    const __nv_bfloat16* expert_codebook) {
  // Four MMA lanes consume four consecutive BF16 pairs. A run of four pairs
  // crosses at most one D6 codeword boundary, so both decoder cases are
  // compile-time constants; only a lane predicate chooses the owner.
  constexpr int kLowGroup = StartPair / 3;
  constexpr int kPairInLowGroup = StartPair % 3;
  constexpr int kFirstHighLane = 3 - kPairInLowGroup;
  const uint32_t low_id =
      decode_packet_id(word0, word1, word2, kLowGroup);
  const uint32_t high_id =
      decode_packet_id(word0, word1, word2, kLowGroup + 1);
  const bool use_high = lane_in_group >= kFirstHighLane;
  const uint32_t id = use_high ? high_id : low_id;
  const int pair_in_codeword = kPairInLowGroup + lane_in_group -
      (use_high ? 3 : 0);
  return load_u32(
      expert_codebook + id * kGroupSize + pair_in_codeword * 2);
}

__device__ __forceinline__ void mma_bf16_m16n8k16(
    float (&accumulator)[4],
    const uint32_t (&a)[4],
    uint32_t b0,
    uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(accumulator[0]),
        "+f"(accumulator[1]),
        "+f"(accumulator[2]),
        "+f"(accumulator[3])
      : "r"(a[0]),
        "r"(a[1]),
        "r"(a[2]),
        "r"(a[3]),
        "r"(b0),
        "r"(b1));
#endif
}

// Safe padded-D8 fallback used for a non-K48 tail (and retained as the
// control path). Keeping it beside the exact core gives every projection the
// same audited tail semantics without materializing dense weights.
template <int BlockM, bool FullPacket>
__device__ __forceinline__ void compute_padded_packet(
    float (&accumulators)[BlockM / 16][2][4],
    const __nv_bfloat16* shared_stage,
    const __nv_bfloat16* expert_codebook,
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
    if constexpr (FullPacket) {
      // Lane 3 duplicates lane 2's address. The warp coalescer merges that
      // transaction, while lanes 0..2 remain the three shuffle sources.
      const int word_lane = lane_in_group < 3 ? lane_in_group : 2;
      const int word = packet * 3 + word_lane;
      owned_words[n_half] = load_assignment_word(
          expert_assignments,
          word,
          output_n,
          assignment_word_stride,
          assignment_output_stride);
    } else if (lane_in_group < 3 && output_n < out_features) {
      const int word = packet * 3 + lane_in_group;
      if (word < num_words) {
        owned_words[n_half] = load_assignment_word(
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
      const uint32_t id0 = decode_packet_id(word0, word1, word2, group0);
      const uint32_t id1 = decode_packet_id(word0, word1, word2, group1);
      if constexpr (FullPacket) {
        // D6 has only three BF16 pairs. Lane 3 repeats lane 2's cache-line
        // request and is then zeroed without a divergent lookup branch.
        const int pair_lane = lane_in_group < 3 ? lane_in_group : 2;
        const int pair = pair_lane * 2;
        const uint32_t active = lane_in_group < 3 ? 0xFFFFFFFFu : 0u;
        b0_by_half[n_half] =
            load_u32(expert_codebook + id0 * kGroupSize + pair) & active;
        b1_by_half[n_half] =
            load_u32(expert_codebook + id1 * kGroupSize + pair) & active;
      } else if (output_n < out_features) {
        const int pair = lane_in_group * 2;
        if (lane_in_group < 3 && packet * 8 + group0 < num_groups &&
            id0 < codebook_size) {
          b0_by_half[n_half] =
              load_u32(expert_codebook + id0 * kGroupSize + pair);
        }
        if (lane_in_group < 3 && packet * 8 + group1 < num_groups &&
            id1 < codebook_size) {
          b1_by_half[n_half] =
              load_u32(expert_codebook + id1 * kGroupSize + pair);
        }
      }
    }

#pragma unroll
    for (int m = 0; m < kMTiles; ++m) {
      uint32_t a[4];
      load_a_fragment(a, shared_stage, m, fragment, lane);
#pragma unroll
      for (int n_half = 0; n_half < 2; ++n_half) {
        mma_bf16_m16n8k16(
            accumulators[m][n_half],
            a,
            b0_by_half[n_half],
            b1_by_half[n_half]);
      }
    }
  }
}

template <int BlockM, int Fragment>
__device__ __forceinline__ void compute_exact_fragment(
    float (&accumulators)[BlockM / 16][2][4],
    const __nv_bfloat16* shared_stage,
    const __nv_bfloat16* expert_codebook,
    const uint32_t (&packet_words)[2][3]) {
  constexpr int kMTiles = BlockM / 16;
  const int lane = threadIdx.x & 31;
  const int lane_in_group = lane & 3;
  uint32_t b0_by_half[2];
  uint32_t b1_by_half[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const uint32_t word0 = packet_words[n_half][0];
    const uint32_t word1 = packet_words[n_half][1];
    const uint32_t word2 = packet_words[n_half][2];
    b0_by_half[n_half] = load_exact_b_operand<Fragment * 8>(
        word0, word1, word2, lane_in_group, expert_codebook);
    b1_by_half[n_half] = load_exact_b_operand<Fragment * 8 + 4>(
        word0, word1, word2, lane_in_group, expert_codebook);
  }

#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    uint32_t a[4];
    load_a_fragment(a, shared_stage, m, Fragment, lane);
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      mma_bf16_m16n8k16(
          accumulators[m][n_half],
          a,
          b0_by_half[n_half],
          b1_by_half[n_half]);
    }
  }
}

template <int BlockM>
__device__ __forceinline__ void compute_exact_k48_packet(
    float (&accumulators)[BlockM / 16][2][4],
    const __nv_bfloat16* shared_stage,
    const __nv_bfloat16* expert_codebook,
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
    const int word = packet * 3 + word_lane;
    owned_words[n_half] = output_n < out_features
        ? load_assignment_word(
              expert_assignments,
              word,
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

  compute_exact_fragment<BlockM, 0>(
      accumulators, shared_stage, expert_codebook, packet_words);
  compute_exact_fragment<BlockM, 1>(
      accumulators, shared_stage, expert_codebook, packet_words);
  compute_exact_fragment<BlockM, 2>(
      accumulators, shared_stage, expert_codebook, packet_words);
}

template <int Group>
__device__ __forceinline__ uint32_t load_codeword_lane_pair(
    uint32_t word0,
    uint32_t word1,
    uint32_t word2,
    int lane_in_group,
    const __nv_bfloat16* expert_codebook) {
  const uint32_t id = decode_packet_id(word0, word1, word2, Group);
  const int pair_lane = lane_in_group < 3 ? lane_in_group : 2;
  return load_u32(expert_codebook + id * kGroupSize + pair_lane * 2);
}

template <int StartPair>
__device__ __forceinline__ uint32_t assemble_codeword_operand(
    uint32_t low_codeword,
    uint32_t high_codeword,
    int lane_in_group) {
  // Inputs are codeword-major: lanes 0..2 own the three contiguous BF16
  // pairs. Move only pairs crossing a K16 boundary inside each four-lane
  // output-column subgroup.
  constexpr int kPairInLowGroup = StartPair % 3;
  constexpr int kFirstHighLane = 3 - kPairInLowGroup;
  const bool use_high = lane_in_group >= kFirstHighLane;
  const int raw_high_source = kPairInLowGroup + lane_in_group - 3;
  const int high_source = raw_high_source > 0 ? raw_high_source : 0;
  const uint32_t high_value = __shfl_sync(
      0xFFFFFFFFu, high_codeword, high_source, 4);
  if constexpr (kPairInLowGroup == 0) {
    return use_high ? high_value : low_codeword;
  } else {
    const int raw_low_source = kPairInLowGroup + lane_in_group;
    const int low_source = raw_low_source < 2 ? raw_low_source : 2;
    const uint32_t low_value = __shfl_sync(
        0xFFFFFFFFu, low_codeword, low_source, 4);
    return use_high ? high_value : low_value;
  }
}

template <int BlockM, int Fragment>
__device__ __forceinline__ void mma_exact_fragment(
    float (&accumulators)[BlockM / 16][2][4],
    const __nv_bfloat16* shared_stage,
    const uint32_t (&b0_by_half)[2],
    const uint32_t (&b1_by_half)[2]) {
  constexpr int kMTiles = BlockM / 16;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    uint32_t a[4];
    load_a_fragment(a, shared_stage, m, Fragment, lane);
#pragma unroll
    for (int n_half = 0; n_half < 2; ++n_half) {
      mma_bf16_m16n8k16(
          accumulators[m][n_half],
          a,
          b0_by_half[n_half],
          b1_by_half[n_half]);
    }
  }
}

template <int BlockM>
__device__ __forceinline__ void compute_coalesced_k48_packet(
    float (&accumulators)[BlockM / 16][2][4],
    const __nv_bfloat16* shared_stage,
    const __nv_bfloat16* expert_codebook,
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
        ? load_assignment_word(
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
    const uint32_t group0 = load_codeword_lane_pair<0>(
        word0, word1, word2, lane_in_group, expert_codebook);
    const uint32_t group1 = load_codeword_lane_pair<1>(
        word0, word1, word2, lane_in_group, expert_codebook);
    b0[n_half] = assemble_codeword_operand<0>(
        group0, group1, lane_in_group);
    carry_group2[n_half] = load_codeword_lane_pair<2>(
        word0, word1, word2, lane_in_group, expert_codebook);
    b1[n_half] = assemble_codeword_operand<4>(
        group1, carry_group2[n_half], lane_in_group);
  }
  mma_exact_fragment<BlockM, 0>(accumulators, shared_stage, b0, b1);

  uint32_t carry_group5[2];
#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const uint32_t word0 = packet_words[n_half][0];
    const uint32_t word1 = packet_words[n_half][1];
    const uint32_t word2 = packet_words[n_half][2];
    const uint32_t group3 = load_codeword_lane_pair<3>(
        word0, word1, word2, lane_in_group, expert_codebook);
    b0[n_half] = assemble_codeword_operand<8>(
        carry_group2[n_half], group3, lane_in_group);
    const uint32_t group4 = load_codeword_lane_pair<4>(
        word0, word1, word2, lane_in_group, expert_codebook);
    carry_group5[n_half] = load_codeword_lane_pair<5>(
        word0, word1, word2, lane_in_group, expert_codebook);
    b1[n_half] = assemble_codeword_operand<12>(
        group4, carry_group5[n_half], lane_in_group);
  }
  mma_exact_fragment<BlockM, 1>(accumulators, shared_stage, b0, b1);

#pragma unroll
  for (int n_half = 0; n_half < 2; ++n_half) {
    const uint32_t word0 = packet_words[n_half][0];
    const uint32_t word1 = packet_words[n_half][1];
    const uint32_t word2 = packet_words[n_half][2];
    const uint32_t group6 = load_codeword_lane_pair<6>(
        word0, word1, word2, lane_in_group, expert_codebook);
    b0[n_half] = assemble_codeword_operand<16>(
        carry_group5[n_half], group6, lane_in_group);
    const uint32_t group7 = load_codeword_lane_pair<7>(
        word0, word1, word2, lane_in_group, expert_codebook);
    b1[n_half] = assemble_codeword_operand<20>(
        group6, group7, lane_in_group);
  }
  mma_exact_fragment<BlockM, 2>(accumulators, shared_stage, b0, b1);
}

}  // namespace nowag::k48_core
