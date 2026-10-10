#include <torch/extension.h>

#include <optional>
#include <vector>

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
    bool preapplied_input_norm);

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
    bool preapplied_input_norm);

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
    bool preapply_down_norm);

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
    bool preapply_down_norm);

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
    bool assignment_l2_only);

std::vector<torch::Tensor> nowag_moe_sparse_route_align_cuda(
    torch::Tensor topk_ids,
    int64_t block_size,
    int64_t num_experts);

void nowag_moe_build_adaptive_tasks_cuda(
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor tasks64,
    torch::Tensor tasks16,
    torch::Tensor task_counts,
    int64_t num_routes,
    int64_t num_experts);

void nowag_moe_build_adaptive_tasks_tail64_cuda(
    torch::Tensor sorted_tickets,
    torch::Tensor expert_ids16,
    torch::Tensor num_tickets_post_padded,
    torch::Tensor tasks64,
    torch::Tensor tasks16,
    torch::Tensor task_counts,
    int64_t num_routes,
    int64_t num_experts);

namespace {

// Keep the original positional extension call valid.  The Python boundary
// uses the overload below whenever a padded physical workspace is requested.
void nowag_moe_gate_up_exact_k48_cuda_legacy(
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
    bool precompute_tokens) {
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
      0,
      true,
      0.0,
      true);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "moe_down_padded64",
      &nowag_moe_down_padded64_cuda,
      "NoWag SM80+ padded-K64 Down lookup-MMA control kernel");
  module.def(
      "moe_down_exact_k48_adaptive",
      &nowag_moe_down_exact_k48_adaptive_cuda,
      "Exact-K48 Down over device-built BM64/BM16 task queues");
  module.def(
      "moe_gate_up_exact_k48",
      &nowag_moe_gate_up_exact_k48_cuda_legacy,
      "NoWag SM80+ Gate/Up using the shared exact-K48 lookup-MMA core");
  module.def(
      "moe_gate_up_exact_k48",
      &nowag_moe_gate_up_exact_k48_cuda,
      "NoWag SM80+ Gate/Up using the shared exact-K48 lookup-MMA core");
  module.def(
      "moe_gate_up_exact_k48_adaptive",
      &nowag_moe_gate_up_exact_k48_adaptive_cuda,
      "Exact-K48 Gate/Up over device-built BM64/BM16 task queues");
  module.def(
      "moe_gate_up_exact_k48_codebook_cache",
      &nowag_moe_gate_up_exact_k48_codebook_cache_cuda,
      "BM16 Gate/Up with assignment/codebook cache controls");
  module.def(
      "moe_sparse_route_align",
      &nowag_moe_sparse_route_align_cuda,
      "Single-CTA route alignment whose work is independent of expert count");
  module.def(
      "moe_build_adaptive_tasks",
      &nowag_moe_build_adaptive_tasks_cuda,
      "Build BM64/BM16 task queues from BM16-aligned routes");
  module.def(
      "moe_build_adaptive_tasks_tail64",
      &nowag_moe_build_adaptive_tasks_tail64_cuda,
      "Build BM64/BM16 queues with dense residuals folded into BM64");
}
