"""NoWAG host-bank descriptor for the common CPU worker pool."""

import torch

from .weights import BANK_NAMES, BIAS_NAMES, _words


def prepare_cpu_weights(math, state, banks, shared):
    from .method import check_nowag_math

    check_nowag_math(math)
    layers = len(banks["gate_input_norm"])
    experts = banks["gate_input_norm"][0].shape[0]
    codebook = shared.get("codebook")
    if codebook is None:
        raise RuntimeError("NoWAG CPU MoE cache has no host codebook")
    d = int(codebook.shape[1]) if codebook.ndim == 2 else 0
    if codebook.dtype != torch.bfloat16 or tuple(codebook.shape) != (4096, d) or d not in (4, 6):
        raise ValueError(
            "NoWAG CPU MoE requires a shared BF16 [4096, D] codebook with D in (4, 6)"
        )
    gate_in = banks["gate_input_norm"]
    gate_out = banks["gate_output_norm"]
    H = int(gate_in[0].shape[1])
    I = int(gate_out[0].shape[1])

    expected = {
        "gate_assignments": (experts, _words(H, d, 12), I),
        "gate_input_norm": (experts, H),
        "gate_output_norm": (experts, I),
        "up_assignments": (experts, _words(H, d, 12), I),
        "up_input_norm": (experts, H),
        "up_output_norm": (experts, I),
        "down_assignments": (experts, _words(I + state.down_start_lane, d, 12), H),
        "down_input_norm": (experts, I),
        "down_output_norm": (experts, H),
        "gate_bias": (experts, I),
        "up_bias": (experts, I),
        "down_bias": (experts, H),
    }
    names = BANK_NAMES + tuple(n for n in BIAS_NAMES if n in banks)
    for name in names:
        want_dtype = torch.int32 if name.endswith("assignments") else torch.bfloat16
        for layer_id, tensor in enumerate(banks[name]):
            if tensor.dtype != want_dtype or tuple(tensor.shape) != expected[name]:
                raise ValueError(
                    f"NoWAG bank {name!r} layer {layer_id} must be "
                    f"{expected[name]} {want_dtype}, got {tuple(tensor.shape)} "
                    f"{tensor.dtype}"
                )
    descriptor = torch.tensor(
        [
            [
                banks[name][layer_id].data_ptr() if name in banks else 0
                for name in BANK_NAMES + BIAS_NAMES
            ]
            for layer_id in range(layers)
        ],
        dtype=torch.int64,
    )
    retained = [descriptor, codebook]
    for name in names:
        retained.extend(banks[name])
    ptrs = dict(
        gate_up_ptr=0,
        down_ptr=0,
        gate_up_scale_ptr=0,
        gate_up_global_ptr=0,
        down_scale_ptr=0,
        down_global_ptr=0,
        gate_up_bias_ptr=0,
        down_bias_ptr=0,
        nowag_bank_table_ptr=descriptor.data_ptr(),
        nowag_codebook_ptr=codebook.data_ptr(),
        nowag_group_size=d,
        nowag_down_start_lane=state.down_start_lane,
        nowag_round_input=math.gate_up_input_rounding is not None,
        nowag_round_middle=math.down_input_rounding is not None,
        nowag_preapply_down_norm=math.down_input_rounding is None,
        nowag_weight_middle=math.router_weight_on_down_input,
    )
    return ptrs, (H, I), retained
