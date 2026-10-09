#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace {

using nvcuda::wmma::accumulator;
using nvcuda::wmma::col_major;
using nvcuda::wmma::fragment;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::row_major;

constexpr int kThreads = 256;
constexpr int kBlockN = 128;
constexpr int kGroupsPerStage = 8;
constexpr int kCodewordWidth = 6;
constexpr int kPaddedCodewordWidth = 8;
constexpr int kPhysicalK = kGroupsPerStage * kPaddedCodewordWidth;
constexpr int kWarps = kThreads / 32;
constexpr int kTraceMetrics = 7;

enum TraceMetric : int {
  kTraceTotal = 0,
  kTraceCodebookIssue = 1,
  kTraceActivationStage = 2,
  kTraceMma = 3,
  kTraceWaitBarrier = 4,
  kTraceProjectionEpilogue = 5,
  kTraceStages = 6,
};

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(
    float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ void cp_async_ca_16(
    void* shared_destination,
    const void* global_source,
    int source_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t shared_address =
      static_cast<uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::
          "r"(shared_address),
      "l"(global_source),
      "r"(source_bytes));
#else
  if (source_bytes == 16) {
    *reinterpret_cast<uint4*>(shared_destination) =
        *reinterpret_cast<const uint4*>(global_source);
  } else {
    *reinterpret_cast<uint4*>(shared_destination) = make_uint4(0, 0, 0, 0);
  }
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

__device__ __forceinline__ uint32_t load_assignment_word(
    const int32_t* assignments,
    int expert,
    int output_row,
    int word,
    int num_words,
    int out_features,
    bool word_major) {
  if (word < 0 || word >= num_words) {
    return 0;
  }
  const int64_t expert_stride =
      static_cast<int64_t>(out_features) * num_words;
  const int64_t offset = word_major
      ? static_cast<int64_t>(expert) * expert_stride +
            static_cast<int64_t>(word) * out_features + output_row
      : static_cast<int64_t>(expert) * expert_stride +
            static_cast<int64_t>(output_row) * num_words + word;
  return static_cast<uint32_t>(assignments[offset]);
}

__device__ __forceinline__ uint32_t decode_generic_12bit(
    const int32_t* assignments,
    int expert,
    int output_row,
    int group,
    int num_words,
    int out_features,
    bool word_major) {
  const int bit = group * 12;
  const int word = bit >> 5;
  const int shift = bit & 31;
  const uint32_t lo = load_assignment_word(
      assignments,
      expert,
      output_row,
      word,
      num_words,
      out_features,
      word_major);
  uint64_t joined = lo;
  if (shift > 20) {
    const uint32_t hi = load_assignment_word(
        assignments,
        expert,
        output_row,
        word + 1,
        num_words,
        out_features,
        word_major);
    joined |= static_cast<uint64_t>(hi) << 32;
  }
  return static_cast<uint32_t>((joined >> shift) & 0xFFFu);
}

__device__ __forceinline__ uint32_t decode_packet_12bit(
    const int32_t* assignments,
    int expert,
    int output_row,
    int packet,
    int lane,
    int num_words,
    int out_features,
    bool word_major) {
  const int word_base = packet * 3;
  uint32_t owned_word = 0;
  if (lane < 3) {
    owned_word = load_assignment_word(
        assignments,
        expert,
        output_row,
        word_base + lane,
        num_words,
        out_features,
        word_major);
  }
  const uint32_t word0 = __shfl_sync(0xFFFFFFFFu, owned_word, 0, 8);
  const uint32_t word1 = __shfl_sync(0xFFFFFFFFu, owned_word, 1, 8);
  const uint32_t word2 = __shfl_sync(0xFFFFFFFFu, owned_word, 2, 8);
  const int bit = lane * 12;
  uint64_t joined;
  int shift;
  if (bit < 32) {
    joined = static_cast<uint64_t>(word0) |
        (static_cast<uint64_t>(word1) << 32);
    shift = bit;
  } else {
    joined = static_cast<uint64_t>(word1) |
        (static_cast<uint64_t>(word2) << 32);
    shift = bit - 32;
  }
  return static_cast<uint32_t>((joined >> shift) & 0xFFFu);
}

template <typename T, int BlockM, bool Trace>
__device__ __forceinline__ void produce_stage(
    int pipe,
    int k_stage,
    T* shared_a,
    T* shared_b,
    const T* hidden,
    const T* codebook,
    const int32_t* assignments,
    const T* input_norm,
    const int32_t* sorted_tickets,
    int expert,
    int block_m_start,
    int block_n_start,
    int num_routes,
    int top_k,
    int hidden_size,
    int out_features,
    int codebook_size,
    int codebook_banks,
    int num_groups,
    int num_words,
    bool word_major,
    bool use_block12_decoder,
    unsigned long long& codebook_issue_cycles,
    unsigned long long& activation_stage_cycles) {
  unsigned long long phase_start = 0;
  if constexpr (Trace) {
    phase_start = clock64();
  }
  T* stage_a = shared_a + pipe * BlockM * kPhysicalK;
  T* stage_b = shared_b + pipe * kBlockN * kPhysicalK;
  const int thread = threadIdx.x;
  const int subgroup = thread >> 3;
  const int lane = thread & 7;

  // Each eight-lane subgroup reconstructs one output row at a time.  One
  // lane owns one already aligned D=8 runtime codeword and copies it directly
  // from HBM into the padded shared-memory record.
#pragma unroll
  for (int row_batch = 0; row_batch < 4; ++row_batch) {
    const int local_n = row_batch * 32 + subgroup;
    const int output_row = block_n_start + local_n;
    const int group = k_stage * kGroupsPerStage + lane;
    const bool valid = output_row < out_features && group < num_groups;
    uint32_t codeword_id = 0;
    if (output_row < out_features) {
      codeword_id = use_block12_decoder
          ? decode_packet_12bit(
                assignments,
                expert,
                output_row,
                k_stage,
                lane,
                num_words,
                out_features,
                word_major)
          : decode_generic_12bit(
                assignments,
                expert,
                output_row,
                group,
                num_words,
                out_features,
                word_major);
    }
    const bool valid_codeword = valid && codeword_id < codebook_size;
    const int codebook_expert = codebook_banks == 1 ? 0 : expert;
    const T* safe_source = codebook + static_cast<int64_t>(codebook_expert) *
        codebook_size * kPaddedCodewordWidth;
    const T* source = valid_codeword
        ? safe_source +
            static_cast<int64_t>(codeword_id) * kPaddedCodewordWidth
        : safe_source;
    T* destination = stage_b +
        (static_cast<int64_t>(local_n) * kPhysicalK + lane *
         kPaddedCodewordWidth);
    const int source_bytes = valid_codeword ? 16 : 0;
    cp_async_ca_16(destination, source, source_bytes);
  }
  cp_async_commit();

  if constexpr (Trace) {
    const unsigned long long codebook_done = clock64();
    codebook_issue_cycles += codebook_done - phase_start;
    phase_start = codebook_done;
  }

  // Construct the normalized activation tile synchronously.  The outstanding
  // codebook copies progress while these multiplies and stores execute.
  constexpr int kActivationElements = BlockM * kPhysicalK;
  for (int index = thread; index < kActivationElements; index += kThreads) {
    const int local_m = index / kPhysicalK;
    const int physical_k = index - local_m * kPhysicalK;
    const int group_in_stage = physical_k / kPaddedCodewordWidth;
    const int lane_in_codeword = physical_k & 7;
    const int group = k_stage * kGroupsPerStage + group_in_stage;
    const int real_k = group * kCodewordWidth + lane_in_codeword;
    const int ticket = sorted_tickets[block_m_start + local_m];
    const bool valid = ticket < num_routes && lane_in_codeword <
        kCodewordWidth && real_k < hidden_size;
    float scaled = 0.0f;
    if (valid) {
      const int token = ticket / top_k;
      scaled = to_float(hidden[static_cast<int64_t>(token) * hidden_size + real_k]) *
          to_float(input_norm[static_cast<int64_t>(expert) * hidden_size + real_k]);
    }
    stage_a[index] = from_float<T>(scaled);
  }
  if constexpr (Trace) {
    activation_stage_cycles += clock64() - phase_start;
  }
}

template <typename T, int BlockM, bool Trace>
__global__ __launch_bounds__(kThreads) void projection_pipeline_kernel(
    const T* hidden,
    const T* codebook,
    const int32_t* assignments,
    const T* input_norm,
    const T* output_norm,
    const int32_t* sorted_tickets,
    const int32_t* expert_ids,
    const int32_t* num_tickets_post_padded,
    T* projection_output,
    int output_rows,
    int num_routes,
    int top_k,
    int hidden_size,
    int out_features,
    int codebook_size,
    int codebook_banks,
    int num_groups,
    int num_words,
    int alignment_block_ratio,
    bool word_major,
    bool use_block12_decoder,
    unsigned long long* trace,
    int trace_projection) {
  const int block_m_index = blockIdx.x;
  const int block_m_start = block_m_index * BlockM;
  const int padded_rows = *num_tickets_post_padded;
  if (block_m_start >= padded_rows) {
    return;
  }
  const int block_n_start = blockIdx.y * kBlockN;
  const int expert = expert_ids[block_m_index / alignment_block_ratio];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  unsigned long long codebook_issue_cycles = 0;
  unsigned long long activation_stage_cycles = 0;
  unsigned long long mma_cycles = 0;
  unsigned long long wait_barrier_cycles = 0;
  unsigned long long projection_epilogue_cycles = 0;
  unsigned long long kernel_start = 0;
  if constexpr (Trace) {
    kernel_start = clock64();
  }

  extern __shared__ __align__(16) unsigned char shared_raw[];
  T* shared_a = reinterpret_cast<T*>(shared_raw);
  T* shared_b = shared_a + 2 * BlockM * kPhysicalK;

  constexpr int kMTiles = BlockM / 16;
  fragment<accumulator, 16, 16, 16, float> accumulators[kMTiles];
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    nvcuda::wmma::fill_fragment(accumulators[m_tile], 0.0f);
  }

  const int k_stages = (num_groups + kGroupsPerStage - 1) /
      kGroupsPerStage;
  produce_stage<T, BlockM, Trace>(
      0,
      0,
      shared_a,
      shared_b,
      hidden,
      codebook,
      assignments,
      input_norm,
      sorted_tickets,
      expert,
      block_m_start,
      block_n_start,
      num_routes,
      top_k,
      hidden_size,
      out_features,
      codebook_size,
      codebook_banks,
      num_groups,
      num_words,
      word_major,
      use_block12_decoder,
      codebook_issue_cycles,
      activation_stage_cycles);
  unsigned long long phase_start = 0;
  if constexpr (Trace) {
    phase_start = clock64();
  }
  cp_async_wait_all();
  __syncthreads();
  if constexpr (Trace) {
    wait_barrier_cycles += clock64() - phase_start;
  }

  int read_pipe = 0;
  for (int k_stage = 0; k_stage < k_stages; ++k_stage) {
    const int next_stage = k_stage + 1;
    const int write_pipe = read_pipe ^ 1;
    if (next_stage < k_stages) {
      produce_stage<T, BlockM, Trace>(
          write_pipe,
          next_stage,
          shared_a,
          shared_b,
          hidden,
          codebook,
          assignments,
          input_norm,
          sorted_tickets,
          expert,
          block_m_start,
          block_n_start,
          num_routes,
          top_k,
          hidden_size,
          out_features,
          codebook_size,
          codebook_banks,
          num_groups,
          num_words,
          word_major,
          use_block12_decoder,
          codebook_issue_cycles,
          activation_stage_cycles);
    }

    T* stage_a = shared_a + read_pipe * BlockM * kPhysicalK;
    T* stage_b = shared_b + read_pipe * kBlockN * kPhysicalK;
    if constexpr (Trace) {
      phase_start = clock64();
    }
#pragma unroll
    for (int k_tile = 0; k_tile < kPhysicalK / 16; ++k_tile) {
      fragment<matrix_b, 16, 16, 16, T, col_major> b_fragment;
      nvcuda::wmma::load_matrix_sync(
          b_fragment,
          stage_b + static_cast<int64_t>(warp * 16) * kPhysicalK +
              k_tile * 16,
          kPhysicalK);
#pragma unroll
      for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
        fragment<matrix_a, 16, 16, 16, T, row_major> a_fragment;
        nvcuda::wmma::load_matrix_sync(
            a_fragment,
            stage_a + static_cast<int64_t>(m_tile * 16) * kPhysicalK +
                k_tile * 16,
            kPhysicalK);
        nvcuda::wmma::mma_sync(
            accumulators[m_tile],
            a_fragment,
            b_fragment,
            accumulators[m_tile]);
      }
    }
    if constexpr (Trace) {
      mma_cycles += clock64() - phase_start;
      phase_start = clock64();
    }

    if (next_stage < k_stages) {
      cp_async_wait_all();
    }
    __syncthreads();
    if constexpr (Trace) {
      wait_barrier_cycles += clock64() - phase_start;
    }
    read_pipe = write_pipe;
  }

  // The mainloop buffers are dead now, so reuse the same shared allocation as
  // an FP32 accumulator tile for a simple, explicit norm-and-round epilogue.
  if constexpr (Trace) {
    phase_start = clock64();
  }
  float* accumulator_tile = reinterpret_cast<float*>(shared_raw);
#pragma unroll
  for (int m_tile = 0; m_tile < kMTiles; ++m_tile) {
    nvcuda::wmma::store_matrix_sync(
        accumulator_tile + static_cast<int64_t>(m_tile * 16) * kBlockN +
            warp * 16,
        accumulators[m_tile],
        kBlockN,
        nvcuda::wmma::mem_row_major);
  }
  __syncthreads();

  constexpr int kOutputElements = BlockM * kBlockN;
  for (int index = threadIdx.x; index < kOutputElements; index += kThreads) {
    const int local_m = index / kBlockN;
    const int local_n = index - local_m * kBlockN;
    const int row = block_m_start + local_m;
    const int output_n = block_n_start + local_n;
    if (row < output_rows && output_n < out_features) {
      float value = 0.0f;
      if (row < padded_rows) {
        value = accumulator_tile[index] * to_float(
            output_norm[static_cast<int64_t>(expert) * out_features + output_n]);
      }
      projection_output[static_cast<int64_t>(row) * out_features + output_n] =
          from_float<T>(value);
    }
  }
  if constexpr (Trace) {
    projection_epilogue_cycles += clock64() - phase_start;
    if (lane == 0) {
      const int num_n_blocks = gridDim.y;
      const int block_linear =
          (trace_projection * gridDim.x + blockIdx.x) * num_n_blocks +
          blockIdx.y;
      unsigned long long* warp_trace = trace +
          (static_cast<int64_t>(block_linear) * kWarps + warp) *
          kTraceMetrics;
      warp_trace[kTraceTotal] = clock64() - kernel_start;
      warp_trace[kTraceCodebookIssue] = codebook_issue_cycles;
      warp_trace[kTraceActivationStage] = activation_stage_cycles;
      warp_trace[kTraceMma] = mma_cycles;
      warp_trace[kTraceWaitBarrier] = wait_barrier_cycles;
      warp_trace[kTraceProjectionEpilogue] = projection_epilogue_cycles;
      warp_trace[kTraceStages] = k_stages;
    }
  }
}

template <typename T>
__global__ void silu_down_norm_epilogue_kernel(
    T* gate_middle,
    const T* up_output,
    const T* down_input_norm,
    const int32_t* expert_ids,
    const int32_t* num_tickets_post_padded,
    int output_rows,
    int out_features,
    int down_block_m) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  const int64_t total = static_cast<int64_t>(output_rows) * out_features;
  if (linear >= total) {
    return;
  }
  const int row = linear / out_features;
  const int n = linear - static_cast<int64_t>(row) * out_features;
  const int padded_rows = *num_tickets_post_padded;
  if (row >= padded_rows) {
    gate_middle[linear] = from_float<T>(0.0f);
    return;
  }
  const int expert = expert_ids[row / down_block_m];
  const float gate = to_float(gate_middle[linear]);
  const float up = to_float(up_output[linear]);
  const float activated = gate / (1.0f + expf(-gate)) * up;
  const T rounded = from_float<T>(activated);
  const float scaled = to_float(rounded) * to_float(
      down_input_norm[static_cast<int64_t>(expert) * out_features + n]);
  gate_middle[linear] = from_float<T>(scaled);
}

template <typename T, int BlockM, bool Trace>
void launch_projection(
    const T* hidden,
    const T* codebook,
    const int32_t* assignments,
    const T* input_norm,
    const T* output_norm,
    const int32_t* sorted_tickets,
    const int32_t* expert_ids,
    const int32_t* num_tickets_post_padded,
    T* output,
    int output_rows,
    int num_m_blocks,
    int num_routes,
    int top_k,
    int hidden_size,
    int out_features,
    int codebook_size,
    int codebook_banks,
    int num_groups,
    int num_words,
    int alignment_block_ratio,
    bool word_major,
    bool use_block12_decoder,
    unsigned long long* trace,
    int trace_projection,
    cudaStream_t stream) {
  const dim3 block(kThreads);
  const dim3 grid(
      num_m_blocks,
      (out_features + kBlockN - 1) / kBlockN);
  constexpr int kPipelineBytes =
      2 * (BlockM + kBlockN) * kPhysicalK * sizeof(T);
  constexpr int kAccumulatorBytes = BlockM * kBlockN * sizeof(float);
  constexpr int kSharedBytes =
      kPipelineBytes > kAccumulatorBytes ? kPipelineBytes : kAccumulatorBytes;
  projection_pipeline_kernel<T, BlockM, Trace><<<
      grid,
      block,
      kSharedBytes,
      stream>>>(
      hidden,
      codebook,
      assignments,
      input_norm,
      output_norm,
      sorted_tickets,
      expert_ids,
      num_tickets_post_padded,
      output,
      output_rows,
      num_routes,
      top_k,
      hidden_size,
      out_features,
      codebook_size,
      codebook_banks,
      num_groups,
      num_words,
      alignment_block_ratio,
      word_major,
      use_block12_decoder,
      trace,
      trace_projection);
}

template <typename T, bool Trace>
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
    int block_m,
    bool word_major_assignments,
    bool use_block12_decoder,
    unsigned long long* trace,
    cudaStream_t stream) {
  const int hidden_size = hidden_states.size(1);
  const int out_features = gate_output_norm.size(1);
  const int output_rows = gate_up_workspace.size(0) / 2;
  const int num_groups = (hidden_size + kCodewordWidth - 1) /
      kCodewordWidth;
  const int num_words = (num_groups * 12 + 31) / 32;
  T* workspace = reinterpret_cast<T*>(gate_up_workspace.data_ptr());
  T* gate_output = workspace;
  T* up_output = workspace + static_cast<int64_t>(output_rows) * out_features;
  int trace_projection = 0;
  auto launch_one = [&](
                        const torch::Tensor& codebook,
                        const torch::Tensor& assignments,
                        const torch::Tensor& input_norm,
                        const torch::Tensor& output_norm,
                        T* destination) {
    if (block_m == 64) {
      launch_projection<T, 64, Trace>(
          reinterpret_cast<const T*>(hidden_states.data_ptr()),
          reinterpret_cast<const T*>(codebook.data_ptr()),
          assignments.data_ptr<int32_t>(),
          reinterpret_cast<const T*>(input_norm.data_ptr()),
          reinterpret_cast<const T*>(output_norm.data_ptr()),
          sorted_tickets.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(),
          num_tickets_post_padded.data_ptr<int32_t>(),
          destination,
          output_rows,
          num_m_blocks,
          num_routes,
          top_k,
          hidden_size,
          out_features,
          codebook.size(1),
          codebook.size(0),
          num_groups,
          num_words,
          alignment_block_ratio,
          word_major_assignments,
          use_block12_decoder,
          trace,
          trace_projection,
          stream);
    } else {
      launch_projection<T, 32, Trace>(
          reinterpret_cast<const T*>(hidden_states.data_ptr()),
          reinterpret_cast<const T*>(codebook.data_ptr()),
          assignments.data_ptr<int32_t>(),
          reinterpret_cast<const T*>(input_norm.data_ptr()),
          reinterpret_cast<const T*>(output_norm.data_ptr()),
          sorted_tickets.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(),
          num_tickets_post_padded.data_ptr<int32_t>(),
          destination,
          output_rows,
          num_m_blocks,
          num_routes,
          top_k,
          hidden_size,
          out_features,
          codebook.size(1),
          codebook.size(0),
          num_groups,
          num_words,
          alignment_block_ratio,
          word_major_assignments,
          use_block12_decoder,
          trace,
          trace_projection,
          stream);
    }
    ++trace_projection;
  };

  launch_one(
      gate_codebook,
      gate_packed_assignments,
      gate_input_norm,
      gate_output_norm,
      gate_output);
  launch_one(
      up_codebook,
      up_packed_assignments,
      up_input_norm,
      up_output_norm,
      up_output);

  const int64_t epilogue_elements =
      static_cast<int64_t>(output_rows) * out_features;
  const int epilogue_blocks =
      static_cast<int>((epilogue_elements + kThreads - 1) / kThreads);
  const int down_block_m = block_m * alignment_block_ratio;
  silu_down_norm_epilogue_kernel<T><<<
      epilogue_blocks,
      kThreads,
      0,
      stream>>>(
      gate_output,
      up_output,
      reinterpret_cast<const T*>(down_input_norm.data_ptr()),
      expert_ids.data_ptr<int32_t>(),
      num_tickets_post_padded.data_ptr<int32_t>(),
      output_rows,
      out_features,
      down_block_m);
}

void check_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
}

}  // namespace

void nowag_moe_gate_up_pipeline_cuda(
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
    bool word_major_assignments,
    bool use_block12_decoder,
    std::optional<torch::Tensor> debug_trace) {
  TORCH_CHECK(block_m == 32 || block_m == 64, "block_m must be 32 or 64");
  TORCH_CHECK(alignment_block_ratio > 0, "alignment_block_ratio must be positive");
  TORCH_CHECK(num_m_blocks > 0, "num_m_blocks must be positive");
  TORCH_CHECK(num_routes > 0 && top_k > 0, "route dimensions must be positive");
  const at::ScalarType dtype = hidden_states.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "CUDA pipeline supports FP16 and BF16 only");
  check_cuda_tensor(hidden_states, "hidden_states", dtype);
  check_cuda_tensor(gate_codebook, "gate_codebook", dtype);
  check_cuda_tensor(gate_input_norm, "gate_input_norm", dtype);
  check_cuda_tensor(gate_output_norm, "gate_output_norm", dtype);
  check_cuda_tensor(up_codebook, "up_codebook", dtype);
  check_cuda_tensor(up_input_norm, "up_input_norm", dtype);
  check_cuda_tensor(up_output_norm, "up_output_norm", dtype);
  check_cuda_tensor(down_input_norm, "down_input_norm", dtype);
  check_cuda_tensor(gate_up_workspace, "gate_up_workspace", dtype);
  check_cuda_tensor(gate_packed_assignments, "gate_packed_assignments", at::kInt);
  check_cuda_tensor(up_packed_assignments, "up_packed_assignments", at::kInt);
  check_cuda_tensor(sorted_tickets, "sorted_tickets", at::kInt);
  check_cuda_tensor(expert_ids, "expert_ids", at::kInt);
  check_cuda_tensor(num_tickets_post_padded, "num_tickets_post_padded", at::kInt);
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [M,K]");
  TORCH_CHECK(gate_codebook.dim() == 3 && gate_codebook.size(2) == 8,
              "gate_codebook must be the aligned runtime [B,C,8] buffer");
  TORCH_CHECK(up_codebook.dim() == 3 && up_codebook.size(2) == 8,
              "up_codebook must be the aligned runtime [B,C,8] buffer");
  TORCH_CHECK(gate_output_norm.dim() == 2, "gate_output_norm must be [E,N]");
  TORCH_CHECK(up_output_norm.sizes() == gate_output_norm.sizes(),
              "Gate and Up output shapes must match");
  const int64_t num_experts = gate_output_norm.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t out_features = gate_output_norm.size(1);
  TORCH_CHECK(num_experts > 0 && out_features > 0,
              "Gate/Up private tensors must contain at least one expert and row");
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
                  torch::IntArrayRef({num_experts, hidden_size}),
              "gate_input_norm must be private [E,K]");
  TORCH_CHECK(up_input_norm.sizes() == gate_input_norm.sizes(),
              "up_input_norm must be private [E,K]");
  TORCH_CHECK(down_input_norm.sizes() == gate_output_norm.sizes(),
              "down_input_norm must be private [E,N]");
  const int64_t num_groups =
      (hidden_size + kCodewordWidth - 1) / kCodewordWidth;
  const int64_t expected_words = (num_groups * 12 + 31) / 32;
  const auto check_assignments = [&](const torch::Tensor& assignments,
                                     const char* name) {
    if (word_major_assignments) {
      TORCH_CHECK(assignments.sizes() == torch::IntArrayRef(
                      {num_experts, expected_words, out_features}),
                  name, " must be private word-major [E,W,N]");
    } else {
      TORCH_CHECK(assignments.sizes() == torch::IntArrayRef(
                      {num_experts, out_features, expected_words}),
                  name, " must be private row-major [E,N,W]");
    }
  };
  check_assignments(gate_packed_assignments, "gate_packed_assignments");
  check_assignments(up_packed_assignments, "up_packed_assignments");
  TORCH_CHECK(gate_up_workspace.dim() == 2, "workspace must be [2*rows,N]");
  TORCH_CHECK(gate_up_workspace.size(0) % 2 == 0,
              "workspace row count must be even");
  TORCH_CHECK(gate_up_workspace.size(1) == gate_output_norm.size(1),
              "workspace N must match projection N");
  TORCH_CHECK(hidden_states.get_device() == gate_up_workspace.get_device(),
              "all tensors must be on the same CUDA device");

  unsigned long long* trace_pointer = nullptr;
  if (debug_trace.has_value()) {
    torch::Tensor& trace_tensor = *debug_trace;
    check_cuda_tensor(trace_tensor, "debug_trace", at::kLong);
    TORCH_CHECK(
        trace_tensor.get_device() == hidden_states.get_device(),
        "debug_trace must be on the same CUDA device");
    const int64_t num_n_blocks =
        (gate_output_norm.size(1) + kBlockN - 1) / kBlockN;
    const int64_t required_trace_values =
        2 * num_m_blocks * num_n_blocks * kWarps * kTraceMetrics;
    TORCH_CHECK(
        trace_tensor.numel() >= required_trace_values,
        "debug_trace needs at least ",
        required_trace_values,
        " int64 values, got ",
        trace_tensor.numel());
    trace_pointer = reinterpret_cast<unsigned long long*>(
        trace_tensor.data_ptr<int64_t>());
  }

  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (dtype == at::kBFloat16) {
    auto launch = [&](auto trace_tag) {
      constexpr bool kTrace = decltype(trace_tag)::value;
      launch_gate_up<__nv_bfloat16, kTrace>(
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
        static_cast<int>(num_routes),
        static_cast<int>(top_k),
        static_cast<int>(num_m_blocks),
        static_cast<int>(alignment_block_ratio),
        static_cast<int>(block_m),
        word_major_assignments,
        use_block12_decoder,
        trace_pointer,
        stream);
    };
    if (debug_trace.has_value()) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  } else {
    auto launch = [&](auto trace_tag) {
      constexpr bool kTrace = decltype(trace_tag)::value;
      launch_gate_up<half, kTrace>(
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
        static_cast<int>(num_routes),
        static_cast<int>(top_k),
        static_cast<int>(num_m_blocks),
        static_cast<int>(alignment_block_ratio),
        static_cast<int>(block_m),
        word_major_assignments,
        use_block12_decoder,
        trace_pointer,
        stream);
    };
    if (debug_trace.has_value()) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
