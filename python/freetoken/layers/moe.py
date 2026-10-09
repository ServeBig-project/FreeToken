import os
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_backend
from freetoken.moe.fused import fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.utils import div_even

from .base import BaseOP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` (offload) or
# ``_run_experts`` (dense) with a precomputed routing instead of going through the
# generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"


class MoELayer(BaseOP):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        weight_format: str = "bf16",
    ):
        super().__init__()

        self.router = None
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.weight_format = weight_format
        # Bound by make_moe_layer (resident) or the engine (offload family and
        # resident formats loaded through load_expert_banks, which also sets the banks).
        self.expert_method = None
        self.expert_banks: dict[str, torch.Tensor] | None = None
        self.expert_shared: dict[str, torch.Tensor] = {}
        # The decode stream's reserved expert scratch (engine); prefill allocates its own.
        self.expert_workspace: dict[str, torch.Tensor] | None = None
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        if allocate_experts:
            self._alloc_resident_experts(intermediate_size_per_partition)

    def _alloc_resident_experts(self, intermediate_size_per_partition: int) -> None:
        """Allocate the resident (in-GPU) expert weights for ``self.weight_format``.

        The resident sibling of the offload bank schemas; ``_resident_banks`` names
        these tensors by the format's bank names for the bound expert method.
        """
        if self.weight_format == "fp8_block":
            # Stacked block-fp8 experts + bf16 per-128x128-block inverse scales.
            # Full (unpartitioned) intermediate size: this layout is TP=1-only.
            from freetoken.kernel.triton.fp8_block_linear import FP8

            blk = 128
            n, i, h = self.num_experts, self.intermediate_size, self.hidden_size
            self.gate_up_proj = torch.empty(n, 2 * i, h, dtype=FP8)
            self.gate_up_scale_inv = torch.empty(
                n, 2 * i // blk, h // blk, dtype=torch.bfloat16
            )
            self.down_proj = torch.empty(n, h, i, dtype=FP8)
            self.down_scale_inv = torch.empty(n, h // blk, i // blk, dtype=torch.bfloat16)
            return
        assert self.weight_format == "bf16", (
            f"no resident expert allocation for weight_format {self.weight_format!r}"
        )
        self.gate_up_proj = torch.empty(
            self.num_experts,
            2 * intermediate_size_per_partition,
            self.hidden_size,
        )
        self.down_proj = torch.empty(
            self.num_experts,
            self.hidden_size,
            intermediate_size_per_partition,
        )

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _resident_banks(self) -> dict[str, torch.Tensor]:
        if self.expert_banks is not None:
            return self.expert_banks
        if self.weight_format == "fp8_block":
            return {
                "gate_up": self.gate_up_proj, "gate_up_scale": self.gate_up_scale_inv,
                "down": self.down_proj, "down_scale": self.down_scale_inv,
            }
        return {"gate_up": self.gate_up_proj, "down": self.down_proj}

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        prefill = get_global_ctx().batch.uses_extend_path
        return self.expert_method.run(
            hidden_states, topk_ids, topk_weights, self._resident_banks(), self.expert_shared,
            workspace=None if prefill else self.expert_workspace,
            prefill=prefill,
            sort_rows=self.num_experts,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def route(self, hidden_states, router_logits):
        if self.router is not None:
            from freetoken.moe.routing import route_experts

            return route_experts(self, hidden_states, router_logits)
        batch = get_global_ctx().batch
        padding = (batch.num_token_non_padded
                   if not batch.uses_extend_path or batch.is_speculative_verify else None)
        return fused_topk(hidden_states, router_logits, self.top_k, self.renormalize,
                          num_token_non_padded=padding)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        if self.router is not None and get_global_ctx().batch.draft_experts is not None:
            return self.routed_forward(hidden_states, *self.route(hidden_states, router_logits))
        if self.weight_format != "bf16":
            # Quantized resident experts: generic softmax router + format kernel.
            # The bf16 path below stays on ctx.moe_backend byte-for-byte.
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            return self._maybe_all_reduce(
                self._resident_gemm(hidden_states, topk_weights, topk_ids)
            )
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        return self._maybe_all_reduce(final_hidden_states)


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
        )
        self.layer_id = layer_id
        self.offload_cache: OffloadMoeCache | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        ctx = get_global_ctx()
        if ctx.batch.uses_extend_path and not ctx.batch.is_speculative_verify:
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.decode_forward(hidden_states, router_logits)
        return self._maybe_all_reduce(final_hidden_states)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        """
        ctx = get_global_ctx()
        if ctx.batch.uses_extend_path and not ctx.batch.is_speculative_verify:
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(self, hidden_states, router_logits=None):
        return self._decode_routed(hidden_states, *self.route(hidden_states, router_logits))

    def prefill_forward(self, hidden_states, router_logits=None):
        return self._prefill_routed(hidden_states, *self.route(hidden_states, router_logits))

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here."""
        cache = self.offload_cache
        assert cache is not None
        if cache.has_resident_prefill_layer(self.layer_id):
            cache.map_prefill_experts(self.layer_id, topk_ids)
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.bank_views(),
                n=cache.decode_cache_size,
                alphas=cache.alphas_for_resident_layer_slots(self.layer_id),
                is_prefill=False,
            )
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
        cost = get_global_ctx().speculative_cost
        if cost is not None:
            phase = cost.phase(get_global_ctx().batch)
            cost.record_routes(phase, self.layer_id, topk_ids)
            if cost.prefetch is not None:
                cost.prefetch.before_layer(phase, self.layer_id)
        cache.ensure_decode_experts(self.layer_id, topk_ids)
        if cost is not None:
            cost.copy_rows[phase, self.layer_id].copy_(cache.num_indices[0])
            cost.copy_events[phase][self.layer_id][0].record()
        cache.copy_missing()
        if cost is not None:
            cost.copy_events[phase][self.layer_id][1].record()
            if phase == 1 and cost.prefetch is not None:
                cost.prefetch.launch(self.layer_id, topk_ids, get_global_ctx().batch)
            cost.gemm_events[phase][self.layer_id][0].record()
        output = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        if cost is not None:
            cost.gemm_events[phase][self.layer_id][1].record()
        return output

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split is device-side elementwise and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- the GPU weights are zeroed for CPU-assigned
        routes and the CPU ids are -1 for GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        raw = topk_ids.clone()  # raw expert ids for the CPU partial
        draft = get_global_ctx().batch.draft_experts is not None
        if draft:
            # Drafts never admit or fetch experts: GPU hits stay on the GPU, misses go
            # to the CPU, and the target's cache policy sees only target routes.
            topk_ids.copy_(cache.slot_for_id[self.layer_id][topk_ids.long()])
        else:
            cache.ensure_experts_hybrid(self.layer_id, topk_ids)  # slot (hit/fetched) or -1
            if cache.collect_stats:
                cache.record_decode_stats_hybrid(self.layer_id)
        on_gpu = topk_ids >= 0

        cpu_ids = torch.where(on_gpu, raw.new_full((), -1), raw)
        cpu_weights = topk_weights
        if cpu_ids.shape[1] != executor.top_k:
            # Native tasks use the target's route stride; a narrower draft pads
            # with skipped routes.
            pad = executor.top_k - cpu_ids.shape[1]
            cpu_ids = torch.nn.functional.pad(cpu_ids, (0, pad), value=-1)
            cpu_weights = torch.nn.functional.pad(topk_weights, (0, pad), value=0)
        pending = executor.decode_submit(
            self.layer_id, hidden_states, cpu_weights, cpu_ids.contiguous())

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        if not draft:
            cache.copy_missing()
        gpu_slots = topk_ids.clamp_min(0)  # -1 -> slot 0 (zero-weighted below)
        gpu_w = torch.where(on_gpu, topk_weights, topk_weights.new_zeros(())).contiguous()
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        cpu_routed = cpu_routed_early if not _HYBRID_OVERLAP else executor.decode_sync(pending)
        return gpu_routed + cpu_routed

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement for the selected scheduling policy.

        Grouped kernels sort logical experts, then map each block to the shared
        pool. Other kernels consume physical slot ids directly.
        """
        cache = self.offload_cache
        assert cache is not None
        resident = cache.has_resident_prefill_layer(self.layer_id)
        if resident or cache.prefill_group_size:
            expert_map = (
                cache.slot_for_id[self.layer_id] if self.expert_method.logical_sort else None
            )
            logical_ids = topk_ids.clone() if expert_map is not None else topk_ids
            if resident:
                cache.map_prefill_experts(self.layer_id, topk_ids)
                alphas = cache.alphas_for_resident_layer_slots(self.layer_id)
            else:
                # Warmup has no resident group; load only the routed experts.
                cache.ensure_experts(self.layer_id, topk_ids)
                cache.copy_missing()
                alphas = cache.alphas_for_slots(self.layer_id)
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                logical_ids,
                views=cache.bank_views(),
                n=self.num_experts if expert_map is not None else cache.decode_cache_size,
                alphas=alphas,
                is_prefill=True,
                expert_map=expert_map,
            )
        if cache.prefill_overlap:
            views = self._wait_prefill_overlap(cache)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(self.num_experts),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        if self.layer_id == 0:
            cache.begin_prefill()
        cache.prefetch_prefill_layer(self.layer_id)
        if cache.prefill_buffer_count > 1:
            cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id)

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
        expert_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the bound expert method over the bank ``views`` the movement step
        produced (registration order); ``topk_ids`` index their rows, or are logical
        ids when ``expert_map`` is given."""
        banks = dict(zip(cache.bank_schema, views))
        if alphas is not None:
            banks["gate_up_alpha"], banks["down_alpha"] = alphas
        return self.expert_method.run(
            hidden_states, topk_ids, topk_weights, banks, cache.shared,
            workspace=None if is_prefill else self.expert_workspace,
            prefill=is_prefill, sort_rows=n, expert_map=expert_map,
        )


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    weight_format: str = "bf16",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    extra_attrs: dict | None = None,
) -> MoELayer:
    """Build the experts layer for ``config.moe_backend`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid, which
    ignore ``weight_format``: the quant format comes from the offload cache) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``extra_attrs`` become instance attributes --
    the seam for per-format scalars the base signature does not carry (e.g.
    ``hidden_act_alpha``/``swiglu_limit``, read back via ``getattr`` by the engine
    and format kernels). ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_backend(config.moe_backend)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    # NoWAG experts come from their own loader (engine), never from this checkpoint.
    nowag = getattr(config, "expert_quant", "none") == "nowag"
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["layer_id"] = layer_id
    elif nowag:
        kwargs.update(weight_format="nowag", allocate_experts=False)
    else:
        kwargs["weight_format"] = weight_format
    layer = layer_cls(**kwargs)
    layer.layer_id = layer_id
    if config.moe_router is not None:
        from freetoken.moe.routing import ROUTERS

        layer.router = ROUTERS[config.moe_router](layer.top_k, layer.renormalize)
    for name, value in (extra_attrs or {}).items():
        setattr(layer, name, value)
    if not offload and not nowag:
        bind_resident_method(layer)
    return layer


def bind_resident_method(layer: MoELayer) -> None:
    """Bind a resident layer's expert method once its format scalars are set."""
    from freetoken.moe.expert_format import ExpertLayout, bind_expert_method, expert_math

    layout = ExpertLayout(
        layer.weight_format, layer.hidden_size, layer.intermediate_size, layer.num_experts
    )
    layer.expert_method = bind_expert_method(
        expert_math(layer), layout, None, device=torch.get_default_device(), backend="fused"
    )
