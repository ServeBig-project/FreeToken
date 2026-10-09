"""Qwen3.8-Flash-Next (model_type qwen4_exp), served text-only: 36 GDN + 12 QSA layers on
four hyper-connection residual streams, one PLE n-gram embedding layer, 512 NVFP4 routed
experts (top-10) plus a gated shared expert."""

from .config import parse_config
from .model import Qwen4ExpForCausalLM
from .weight import (
    iter_weights,
    load_nvfp4_expert_sources,
    load_nvfp4_expert_sources_parallel,
    load_ple_table,
)

__all__ = [
    "Qwen4ExpForCausalLM",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
    "parse_config",
]
