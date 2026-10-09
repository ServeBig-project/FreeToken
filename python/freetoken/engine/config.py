from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


_LEGACY_SD_CONTROLS = "SD residency, cost, missing-expert loading and prefetch require legacy batching"

@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # None: 4, or with --runtime-cache-gib what the runtime and execution budgets can run.
    max_running_req: int | None = None
    # None: try SD at 4 steps when the components and budgets support it, else AR.
    speculative_num_steps: int | None = None
    # Where layered batching may run SD: outside prefill waves, in both, or only inside.
    speculative_phase: str = "outwave"
    speculative_draft_model_path: str | None = None
    speculative_draft_experts: int = 3
    speculative_draft_residency: str = "off"
    speculative_adaptive_cost: bool = False
    # DFlash: windowed drafter layers keep only their window on the GPU (same attention math);
    # a positive window also bounds the history the drafter's full-attention layers read.
    dflash_compact_kv: bool = True
    dflash_attention_window: int = 0
    # Adaptive DFlash computes its decisions but runs the configured length (overhead A/B).
    dflash_adaptive_observe_only: bool = False
    speculative_draft_load_missing: bool = False
    speculative_verify_prefetch: bool = False
    # GDN ReplaySSM: target AR, draft and verify read checkpoint + per-position update records.
    enable_gdn_replayssm: bool = False
    gdn_replay_buffer_len: int = 32  # records per active request (ring), incl. the draft/verify tail
    # Bytes for all GDN state storage (full states, records, conv windows); None keeps the
    # replay-off pool's bytes.
    gdn_state_budget_bytes: int | None = None
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
    # KV floor for --moe-cache-auto; small by design (MoE-priority). None: 8192.
    kv_reserve_tokens: int | None = None
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
    # Hybrid GDN models default to a prefix tree that also keeps GDN states (cross-request reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots. None: 2.0.
    linear_state_cache_ratio: float | None = None
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
    # Host memory (GiB) for cold prefix-cache data per engine worker; 0 keeps it off.
    prefix_cache_host_gib: float = 0.0
    # GiB per GPU worker shared by target KV, GDN states and records, and drafter history;
    # None keeps each its own fixed pool.
    runtime_cache_gib: float | None = None

    def __post_init__(self) -> None:
        if self.runtime_cache_gib is not None:
            if self.runtime_cache_gib <= 0:
                raise ValueError(f"--runtime-cache-gib must be > 0, got {self.runtime_cache_gib}")
            fixed = {"--num-pages": self.num_page_override, "--num-tokens": self.num_token_override,
                     "--gdn-state-budget-bytes": self.gdn_state_budget_bytes,
                     "--kv-reserve-tokens": self.kv_reserve_tokens,
                     "linear_state_cache_ratio": self.linear_state_cache_ratio}
            given = [name for name, value in fixed.items() if value is not None]
            if given:
                raise ValueError(f"--runtime-cache-gib shares one budget; it conflicts with the "
                                 f"fixed pool sizes {', '.join(given)}")
        if self.prefix_cache_host_gib < 0:
            raise ValueError(
                f"--prefix-cache-host-gib must be >= 0, got {self.prefix_cache_host_gib}")
        if self.speculative_phase not in ("outwave", "all", "inwave"):
            raise ValueError("--speculative-phase must be outwave, all or inwave")
        ring = self.gdn_replay_buffer_len
        if ring < 4 or ring & (ring - 1):
            raise ValueError("--gdn-replay-buffer-len must be a power of two >= 4")
        if self.gdn_state_budget_bytes is not None and self.gdn_state_budget_bytes <= 0:
            raise ValueError("--gdn-state-budget-bytes must be positive")
        if self.speculative_num_steps is None:
            return  # the engine resolves it against the model's components, then validates
        # An explicit 0 turns SD off even beside a draft path, which is then ignored.
        external_draft = self.speculative_draft_model_path is not None and self.speculative_num_steps != 0
        if self.dflash_attention_window < 0:
            raise ValueError("--dflash-attention-window must be >= 0")
        # With SD off the DFlash-only settings are ignored along with the draft path.
        if self.dflash_attention_window and self.speculative_num_steps and not external_draft:
            raise ValueError("--dflash-attention-window requires --speculative-draft-model-path")
        if (self.dflash_adaptive_observe_only and self.speculative_num_steps
                and not (external_draft and self.speculative_adaptive_cost)):
            raise ValueError("--dflash-adaptive-observe-only requires --speculative-draft-model-path "
                             "and --speculative-adaptive-cost")
        if external_draft:
            if not 1 <= self.speculative_num_steps <= 8:
                raise ValueError("DFlash requires 1..8 draft tokens")
            if (self.speculative_draft_residency != "off" or self.speculative_draft_load_missing
                    or self.speculative_verify_prefetch):
                raise ValueError("DFlash does not use target-expert residency, missing loads or route prefetch")
            if self.dtype != torch.bfloat16 or self.page_size != 1:
                raise ValueError("DFlash requires BF16 and page size 1")
        if self.speculative_draft_residency not in ("off", "router"):
            raise ValueError("speculative_draft_residency must be off or router")
        if self.speculative_draft_residency != "off" and self.speculative_num_steps <= 0:
            raise ValueError("--speculative-draft-residency requires SD enabled")
        if not 0 <= self.speculative_num_steps <= 8:
            raise ValueError("--speculative-num-steps must be 0 (off) or 1..8")
        if self.speculative_draft_experts < 1:
            raise ValueError("speculative_draft_experts must be >= 1")
        if self.enable_gdn_replayssm and ring < self.speculative_num_steps + 1:
            raise ValueError(
                f"--gdn-replay-buffer-len {ring} cannot hold a verify window of "
                f"{self.speculative_num_steps + 1} inputs"
            )
        if (self.speculative_adaptive_cost or self.speculative_draft_load_missing
                or self.speculative_verify_prefetch):
            if not 1 <= self.speculative_num_steps <= 8:
                raise ValueError("SD cost, missing-expert loading and prefetch require 1..8 draft steps")
            if self.speculative_draft_load_missing and self.speculative_draft_residency != "router":
                raise ValueError("--speculative-draft-load-missing requires --speculative-draft-residency router")
            model = self.model_config
            # Measured costs need no particular expert format; the expert-loading controls
            # read BF16 expert rows.
            formats = ("none",) if (self.speculative_draft_load_missing
                                   or self.speculative_verify_prefetch) else ("none", "nvfp4")
            if (self.moe_backend not in ("auto", "offload") or self.dtype != torch.bfloat16
                    or model.expert_quant not in formats or self.nowag_expert_path
                    or model.moe_weight_format not in (None, "bf16")):
                raise ValueError("SD controls require --moe-backend offload with BF16 activations; "
                                 "missing-expert loads and prefetch also require BF16 experts")
        if not self.speculative_num_steps:
            return
        if self.tp_info.size != 1:
            raise ValueError("self-speculative decoding requires a single GPU (tp_size=1)")
        policy = getattr(self, "batching_policy", "legacy")
        if policy not in ("auto", "legacy", "layered-pipeline"):
            raise ValueError("speculative decoding requires --batching-policy legacy or layered-pipeline")
        if policy == "legacy" and self.speculative_phase != "outwave":
            raise ValueError(f"--speculative-phase {self.speculative_phase} requires layered-pipeline batching")
        if policy == "layered-pipeline" and self.legacy_sd_controls:
            raise ValueError(_LEGACY_SD_CONTROLS)
        from freetoken.attention.base import AttnType
        from freetoken.moe.routing import ROUTERS

        model = self.model_config
        if not external_draft and (not model.num_experts or model.moe_router not in ROUTERS):
            raise ValueError("self-speculative decoding requires a shared MoE router component")
        unsupported = {model.attn_type_for_layer(i) for i in range(model.num_layers)} - {
            AttnType.FULL, AttnType.LINEAR,
        }
        if unsupported:
            raise ValueError(f"self-speculative state handling is unavailable for {unsupported}")
        if not external_draft and self.speculative_draft_experts > self.model_config.num_experts_per_tok:
            raise ValueError(
                "speculative_draft_experts must not exceed the target's experts per token "
                f"({self.model_config.num_experts_per_tok})"
            )
        if self.moe_backend == "cpu" or self.moe_cpu_layers:
            raise ValueError(
                "speculative decoding does not support all-CPU expert layers; use "
                "--moe-backend offload, hybrid or fused without --moe-cpu-layers"
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
    def legacy_sd_controls(self) -> bool:
        """Whether an SD control that only legacy batching runs is requested."""
        return bool(self.speculative_draft_residency != "off" or self.speculative_adaptive_cost
                    or self.speculative_draft_load_missing or self.speculative_verify_prefetch)

    @property
    def speculative_graphs(self) -> bool:
        from freetoken.attention import attention_backend_info

        backend = self.attention_backend
        graph_attention = (backend != "auto" and "," not in backend
                           and attention_backend_info(backend).speculative_graphs)
        return bool(
            0 < self.speculative_num_steps <= 8
            and self.dtype == torch.bfloat16
            and self.model_config.expert_quant in ("none", "nvfp4") and not self.nowag_expert_path
            and self.model_config.moe_weight_format in (None, "bf16")
            and graph_attention and self.moe_backend in ("offload", "hybrid")
            and self.page_size == 1 and self.tp_info.size == 1
            and getattr(self, "batching_policy", "legacy") in ("legacy", "layered-pipeline")
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
