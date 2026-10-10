#pragma once

// ---------------------------------- NoWAG ----------------------------------
// D4/B12 and D6/B12 assignments stay in runtime [word, output] order. Eight
// 12-bit ids span three words, so AVX2 decodes eight output rows at once and
// gathers the shared BF16 codebook (D lanes per entry) without materializing
// dense experts.
constexpr int NOWAG_ASSIGNMENT_BITS = 12;
constexpr int NOWAG_IDS_PER_BLOCK = 8;
constexpr int NOWAG_WORDS_PER_BLOCK = 3;
constexpr int NOWAG_ACCUMULATORS = 4;
constexpr uint32_t NOWAG_ID_MASK = (1u << NOWAG_ASSIGNMENT_BITS) - 1;

inline int nowag_num_groups(int K, int D) {
  return (K + D - 1) / D;
}

inline int nowag_assignment_words(int K, int D) {
  return (nowag_num_groups(K, D) * NOWAG_ASSIGNMENT_BITS + 31) / 32;
}

using nowag_gemv_fn = void (*)(float*, const uint32_t*, const bf16_t*,
                               const bf16_t*, int, int, int, int, int, int);
using nowag_scale_fn = void (*)(const bf16_t*, const bf16_t*, bf16_t*, int);

inline uint32_t nowag_decode_id(const uint32_t* assignments, int W, int N,
                                int group, int output) {
  const int bit = group * NOWAG_ASSIGNMENT_BITS;
  const int word = bit >> 5;
  const int shift = bit & 31;
  uint64_t packed = assignments[(size_t)word * N + output];
  if (shift + NOWAG_ASSIGNMENT_BITS > 32 && word + 1 < W)
    packed |= (uint64_t)assignments[(size_t)(word + 1) * N + output] << 32;
  return static_cast<uint32_t>((packed >> shift) & NOWAG_ID_MASK);
}

template <int D>
void nowag_gemv_scalar(float* out, const uint32_t* assignments,
                       const bf16_t* x, const bf16_t* codebook, int K,
                       int N, int W, int n0, int n1, int start_lane) {
  const int groups = nowag_num_groups(K + start_lane, D);
  for (int n = n0; n < n1; ++n) {
    float acc[NOWAG_ACCUMULATORS] = {};
    for (int g = 0; g < groups; ++g) {
      const bf16_t* cb = codebook + (size_t)nowag_decode_id(
          assignments, W, N, g, n) * D;
      const int k0 = g * D - start_lane;
      const int lanes = std::min(D, K - k0);
      for (int d = std::max(0, -k0); d < lanes; ++d)
        acc[(k0 + d) & (NOWAG_ACCUMULATORS - 1)] +=
            bf16_to_f32(x[k0 + d]) * bf16_to_f32(cb[d]);
    }
    out[n - n0] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  }
}

void nowag_scale_bf16_scalar(const bf16_t* x, const bf16_t* scale,
                             bf16_t* out, int K) {
  for (int k = 0; k < K; ++k)
    out[k] = f32_to_bf16(bf16_to_f32(x[k]) * bf16_to_f32(scale[k]));
}

#if CPU_MOE_X86
struct NowagAvxAccumulators {
  __m256 values[NOWAG_ACCUMULATORS];
};

template <int D>
__attribute__((target("avx2,fma")))
static inline void nowag_accumulate_codeword(
    NowagAvxAccumulators& acc, __m256i ids, const bf16_t* x,
    const bf16_t* codebook, int lanes, int input_offset) {
  const __m256i base = _mm256_mullo_epi32(
      ids, _mm256_set1_epi32(D / 2));
  const __m256i high_mask = _mm256_set1_epi32((int)0xFFFF0000u);
  for (int d = 0; d < lanes; d += 2) {
    const __m256i indices = _mm256_add_epi32(
        base, _mm256_set1_epi32(d >> 1));
    const __m256i pair = _mm256_i32gather_epi32(
        reinterpret_cast<const int*>(codebook), indices, 4);
    const __m256 low = _mm256_castsi256_ps(_mm256_slli_epi32(pair, 16));
    const int low_acc = (input_offset + d) & (NOWAG_ACCUMULATORS - 1);
    acc.values[low_acc] = _mm256_fmadd_ps(
        low, _mm256_set1_ps(bf16_to_f32(x[d])), acc.values[low_acc]);
    if (d + 1 < lanes) {
      const __m256 high = _mm256_castsi256_ps(
          _mm256_and_si256(pair, high_mask));
      const int high_acc = (input_offset + d + 1) & (NOWAG_ACCUMULATORS - 1);
      acc.values[high_acc] = _mm256_fmadd_ps(
          high, _mm256_set1_ps(bf16_to_f32(x[d + 1])), acc.values[high_acc]);
    }
  }
}

__attribute__((target("avx2,fma")))
static inline __m256i nowag_decode_ids_avx2(const uint32_t* assignments,
                                             int N, int group, int output) {
  const int bit = group * NOWAG_ASSIGNMENT_BITS;
  const int word = bit >> 5;
  const int shift = bit & 31;
  const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      assignments + (size_t)word * N + output));
  __m256i ids = _mm256_srlv_epi32(lo, _mm256_set1_epi32(shift));
  if (shift + NOWAG_ASSIGNMENT_BITS > 32) {
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
        assignments + (size_t)(word + 1) * N + output));
    ids = _mm256_or_si256(
        ids, _mm256_sllv_epi32(hi, _mm256_set1_epi32(32 - shift)));
  }
  return _mm256_and_si256(ids, _mm256_set1_epi32(NOWAG_ID_MASK));
}

template <int D>
__attribute__((target("avx2,fma")))
static inline void nowag_accumulate_eight_groups(
    NowagAvxAccumulators& acc, const uint32_t* assignments, const bf16_t* x,
    const bf16_t* codebook, int N, int group, int output, int start_lane) {
  const int word = (group / NOWAG_IDS_PER_BLOCK) * NOWAG_WORDS_PER_BLOCK;
  const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      assignments + (size_t)word * N + output));
  const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      assignments + (size_t)(word + 1) * N + output));
  const __m256i w2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
      assignments + (size_t)(word + 2) * N + output));
  const __m256i mask = _mm256_set1_epi32(NOWAG_ID_MASK);
  __m256i ids = _mm256_and_si256(w0, mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x, codebook, D, group * D - start_lane);
  ids = _mm256_and_si256(_mm256_srli_epi32(w0, 12), mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x + D, codebook, D,
      (group + 1) * D - start_lane);
  ids = _mm256_and_si256(_mm256_or_si256(
      _mm256_srli_epi32(w0, 24), _mm256_slli_epi32(w1, 8)), mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x + 2 * D, codebook, D,
      (group + 2) * D - start_lane);
  ids = _mm256_and_si256(_mm256_srli_epi32(w1, 4), mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x + 3 * D, codebook, D,
      (group + 3) * D - start_lane);
  ids = _mm256_and_si256(_mm256_srli_epi32(w1, 16), mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x + 4 * D, codebook, D,
      (group + 4) * D - start_lane);
  ids = _mm256_and_si256(_mm256_or_si256(
      _mm256_srli_epi32(w1, 28), _mm256_slli_epi32(w2, 4)), mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x + 5 * D, codebook, D,
      (group + 5) * D - start_lane);
  ids = _mm256_and_si256(_mm256_srli_epi32(w2, 8), mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x + 6 * D, codebook, D,
      (group + 6) * D - start_lane);
  ids = _mm256_and_si256(_mm256_srli_epi32(w2, 20), mask);
  nowag_accumulate_codeword<D>(
      acc, ids, x + 7 * D, codebook, D,
      (group + 7) * D - start_lane);
}

template <int D>
__attribute__((target("avx2,fma")))
void nowag_gemv_avx2(float* out, const uint32_t* assignments,
                     const bf16_t* x, const bf16_t* codebook, int K,
                     int N, int W, int n0, int n1, int start_lane) {
  const int groups = nowag_num_groups(K + start_lane, D);
  int n = n0;
  for (; n + NOWAG_IDS_PER_BLOCK <= n1; n += NOWAG_IDS_PER_BLOCK) {
    NowagAvxAccumulators acc;
    for (int i = 0; i < NOWAG_ACCUMULATORS; ++i)
      acc.values[i] = _mm256_setzero_ps();
    int g = 0;
    if (start_lane) {
      const __m256i ids = nowag_decode_ids_avx2(assignments, N, g++, n);
      for (int lane = start_lane; lane < std::min(D, K + start_lane); ++lane) {
        const __m256i indices = _mm256_add_epi32(
            _mm256_mullo_epi32(ids, _mm256_set1_epi32(D / 2)),
            _mm256_set1_epi32(lane / 2));
        const __m256i pair = _mm256_i32gather_epi32(
            reinterpret_cast<const int*>(codebook), indices, 4);
        const __m256 value = _mm256_castsi256_ps(lane & 1
            ? _mm256_and_si256(pair, _mm256_set1_epi32((int)0xFFFF0000u))
            : _mm256_slli_epi32(pair, 16));
        const int k = lane - start_lane;
        auto& acc_k = acc.values[k & (NOWAG_ACCUMULATORS - 1)];
        acc_k = _mm256_fmadd_ps(value, _mm256_set1_ps(bf16_to_f32(x[k])), acc_k);
      }
    }
    // The block decoder starts on an eight-codeword boundary.
    for (; g % NOWAG_IDS_PER_BLOCK && g < groups; ++g) {
      const int k = g * D - start_lane;
      nowag_accumulate_codeword<D>(acc, nowag_decode_ids_avx2(assignments, N, g, n),
                                   x + k, codebook, std::min(D, K - k), k);
    }
    for (; g + NOWAG_IDS_PER_BLOCK <= groups &&
           (g + NOWAG_IDS_PER_BLOCK) * D - start_lane <= K;
         g += NOWAG_IDS_PER_BLOCK)
      nowag_accumulate_eight_groups<D>(
          acc, assignments, x + g * D - start_lane,
          codebook, N, g, n, start_lane);
    for (; g < groups; ++g) {
      const __m256i ids = nowag_decode_ids_avx2(assignments, N, g, n);
      nowag_accumulate_codeword<D>(
          acc, ids, x + g * D - start_lane, codebook,
          std::min(D, K - g * D + start_lane),
          g * D - start_lane);
    }
    const __m256 sum01 = _mm256_add_ps(acc.values[0], acc.values[1]);
    const __m256 sum23 = _mm256_add_ps(acc.values[2], acc.values[3]);
    _mm256_storeu_ps(out + (n - n0), _mm256_add_ps(sum01, sum23));
  }
  if (n < n1)
    nowag_gemv_scalar<D>(
        out + (n - n0), assignments, x, codebook, K, N, W, n, n1, start_lane);
}

__attribute__((target("avx2,fma")))
void nowag_scale_bf16_avx2(const bf16_t* x, const bf16_t* scale,
                           bf16_t* out, int K) {
  int k = 0;
  const __m256i bias = _mm256_set1_epi32(0x7FFF);
  const __m256i one = _mm256_set1_epi32(1);
  for (; k + 8 <= K; k += 8) {
    const __m128i xi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + k));
    const __m128i ni = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scale + k));
    const __m256 xf = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(xi), 16));
    const __m256 nf = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(ni), 16));
    __m256i bits = _mm256_castps_si256(_mm256_mul_ps(xf, nf));
    bits = _mm256_add_epi32(
        bits, _mm256_add_epi32(bias, _mm256_and_si256(
            _mm256_srli_epi32(bits, 16), one)));
    bits = _mm256_srli_epi32(bits, 16);
    const __m128i packed = _mm_packus_epi32(
        _mm256_castsi256_si128(bits), _mm256_extracti128_si256(bits, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + k), packed);
  }
  nowag_scale_bf16_scalar(x + k, scale + k, out + k, K - k);
}
#endif

template <int D>
nowag_gemv_fn select_nowag_gemv_d() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX2) return nowag_gemv_avx2<D>;
#endif
  (void)t;
  return nowag_gemv_scalar<D>;
}

nowag_gemv_fn select_nowag_gemv(int D) {
  if (D == 4) return select_nowag_gemv_d<4>();
  if (D == 6) return select_nowag_gemv_d<6>();
  throw std::runtime_error("NoWAG CPU MoE supports D4/B12 and D6/B12 codebooks");
}

nowag_scale_fn select_nowag_scale() {
  const IsaTier t = pick_isa();
#if CPU_MOE_X86
  if (t >= ISA_AVX2) return nowag_scale_bf16_avx2;
#endif
  (void)t;
  return nowag_scale_bf16_scalar;
}

