from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import ceil
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    speculative_num_steps: int = 0
    speculative_draft_experts: int = 3
    speculative_draft_residency: str = "off"
    moe_resident_experts: str | None = None
    moe_expert_profile: str | None = None
    speculative_adaptive_profile: str | None = None
    speculative_reuse_expert_cap: int = 0
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    # Expert-only NoWAG output. The base model path still supplies every
    # non-routed-expert weight.
    nowag_expert_path: str | None = None
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    # Offline Engine/Scheduler/LLM callers choose this port when several independent
    # processes run on one host. ServerArgs overrides distributed_addr with server_port + 1.
    distributed_port: int = 2333
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None

    def __post_init__(self) -> None:
        if self.speculative_draft_residency not in ("off", "router", "affinity"):
            raise ValueError("speculative_draft_residency must be off, router, or affinity")
        if self.speculative_draft_residency != "off" and self.speculative_num_steps <= 0:
            raise ValueError("--speculative-draft-residency requires SD enabled")
        if self.speculative_num_steps < 0:
            raise ValueError("speculative_num_steps must be >= 0")
        if self.speculative_draft_experts < 1:
            raise ValueError("speculative_draft_experts must be >= 1")
        if self.speculative_reuse_expert_cap < 0:
            raise ValueError("speculative_reuse_expert_cap must be >= 0")
        if self.speculative_reuse_expert_cap:
            if not self.speculative_num_steps:
                raise ValueError("--speculative-reuse-expert-cap requires SD enabled")
            model = self.model_config
            if not model.num_experts_per_tok <= self.speculative_reuse_expert_cap <= model.num_experts:
                raise ValueError("reuse expert cap must be between target top-k and experts per layer")
        if self.speculative_adaptive_profile:
            if not self.speculative_num_steps:
                raise ValueError("--speculative-adaptive-profile requires SD enabled")
            self.draft_cost
        if not (self.speculative_num_steps or self.moe_resident_experts or self.moe_expert_profile):
            return
        if self.tp_info.size != 1:
            raise ValueError("self-speculative decoding requires a single GPU (tp_size=1)")
        if getattr(self, "batching_policy", "legacy") != "legacy":
            raise ValueError("self-speculative decoding requires --batching-policy legacy")
        if self.hf_config.architectures[0] != "Qwen3MoeForCausalLM":
            raise ValueError("self-speculative decoding currently supports only Qwen3 MoE")
        if self.speculative_num_steps and self.speculative_draft_experts > self.model_config.num_experts_per_tok:
            raise ValueError(
                "speculative_draft_experts must not exceed the target's experts per token "
                f"({self.model_config.num_experts_per_tok})"
            )
        if self.moe_backend in ("cpu", "hybrid") or self.moe_cpu_layers:
            raise ValueError(
                "self-speculative decoding requires GPU expert execution; use "
                "--moe-backend offload or fused without --moe-cpu-layers"
            )
        if self.moe_expert_profile and self.speculative_num_steps:
            raise ValueError("--moe-expert-profile requires ordinary target serving (SD disabled)")
        if self.speculative_draft_residency == "affinity" and self.moe_backend != "fused":
            model = self.model_config
            if (self.nowag_expert_path or model.expert_quant != "none"
                    or model.moe_weight_format not in (None, "bf16")):
                raise ValueError("affinity draft residency requires unquantized floating-point expert weights")
        if self.moe_resident_experts:
            if self.moe_backend == "fused":
                raise ValueError("fused experts are already resident; omit --moe-resident-experts")
            from freetoken.moe.profile import validate_resident_capacity
            count = len(self.resident_experts)
            model = self.model_config
            if not self.moe_cache_auto:
                size = self.moe_cache_size
                if self.moe_cache_rate is not None:
                    size = ceil(model.num_moe_layers * model.num_experts * self.moe_cache_rate)
                if size or self.moe_cache_rate is not None:
                    validate_resident_capacity(size, model.num_experts, count, self.moe_prefill_overlap)

    @cached_property
    def draft_cost(self) -> dict[str, float] | None:
        if self.speculative_adaptive_profile is None:
            return None
        from .speculative_policy import load_draft_cost
        return load_draft_cost(self.speculative_adaptive_profile)

    @cached_property
    def resident_experts(self) -> tuple[tuple[int, int], ...]:
        if self.moe_resident_experts is None:
            return ()
        from freetoken.moe.profile import load_resident_experts
        return load_resident_experts(
            self.moe_resident_experts, self.model_config.num_moe_layers, self.model_config.num_experts
        )

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        return parse_config(self.hf_config)

    @property
    def speculative_graphs(self) -> bool:
        return bool(
            0 < self.speculative_num_steps <= 4 and self.speculative_draft_experts == 3
            and self.speculative_draft_residency in ("off", "router")
            and not self.speculative_adaptive_profile and not self.speculative_reuse_expert_cap
            and not self.resident_experts and not self.moe_expert_profile
            and self.dtype == torch.bfloat16 and self.model_config.model_type == "qwen3_moe"
            and self.model_config.expert_quant == "none" and not self.nowag_expert_path
            and self.model_config.moe_weight_format in (None, "bf16")
            and self.attention_backend == "fi" and self.moe_backend == "offload"
            and self.page_size == 1 and self.tp_info.size == 1
            and getattr(self, "batching_policy", "legacy") == "legacy"
            and self.cuda_graph_max_bs != 0 and self.cuda_graph_bs != []
        )

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.distributed_port}"
