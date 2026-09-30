"""Independent NumPy float64 equations from the public DFlash contract."""

import numpy as np


def ideal(values):
    return values


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


def rms(values, weight, eps, rounding=ideal):
    normalized = values / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + eps)
    return rounding(rounding(normalized) * weight)


def rope(values, positions, theta, rounding=ideal):
    half = values.shape[-1] // 2
    phase = positions[:, None] * theta ** (-np.arange(half, dtype=np.float64) / half)
    cosine = rounding(np.concatenate((np.cos(phase), np.cos(phase)), axis=-1)[:, None, :])
    sine = rounding(np.concatenate((np.sin(phase), np.sin(phase)), axis=-1)[:, None, :])
    rotated = np.concatenate((-values[..., half:], values[..., :half]), axis=-1)
    return rounding(rounding(values * cosine) + rounding(rotated * sine))


def projection(values, positions, layer, heads, head_dim, eps, theta, query=False, rounding=ideal):
    key = "q" if query else "k"
    projected = rounding(values @ layer[key].T).reshape(-1, heads, head_dim)
    return rope(rms(projected, layer[key + "_norm"], eps, rounding), positions, theta, rounding)


def project_context(features, positions, parameters, config, rounding=ideal):
    context = rms(rounding(features @ parameters["fc"].T), parameters["hidden_norm"], config["rms_norm_eps"], rounding)
    projected = []
    for layer in parameters["layers"]:
        key = projection(context, positions, layer, config["num_key_value_heads"],
                         config["head_dim"], config["rms_norm_eps"], config["rope_parameters"]["rope_theta"], rounding=rounding)
        value = rounding(context @ layer["v"].T).reshape(-1, config["num_key_value_heads"], config["head_dim"])
        projected.append((key, value))
    return projected


def attention(queries, keys, values, positions, request_ids, history, mode):
    causal, window_left = mode
    output = np.empty_like(queries)
    group_size = queries.shape[1] // keys.shape[1]
    for row, (request_id, position) in enumerate(zip(request_ids, positions)):
        own = request_ids == request_id
        old_positions, old_keys, old_values = history[request_id]
        all_positions = np.concatenate((old_positions, positions[own]))
        all_keys = np.concatenate((old_keys, keys[own]))
        all_values = np.concatenate((old_values, values[own]))
        visible = np.ones(len(all_positions), dtype=bool)
        if causal:
            visible &= all_positions <= position
        if window_left >= 0:
            visible &= all_positions >= position - window_left
        for head in range(queries.shape[1]):
            kv_head = head // group_size
            scores = all_keys[visible, kv_head] @ queries[row, head] / np.sqrt(queries.shape[-1])
            probability = np.exp(scores - np.max(scores))
            probability /= probability.sum()
            output[row, head] = probability @ all_values[visible, kv_head]
    return output


def forward(embeddings, positions, request_ids, histories, parameters, config, modes, rounding=ideal):
    hidden = embeddings.copy()
    trace = []
    eps = config["rms_norm_eps"]
    for index, layer in enumerate(parameters["layers"]):
        normalized = rms(hidden, layer["input_norm"], eps, rounding)
        query = projection(normalized, positions, layer, config["num_attention_heads"],
                           config["head_dim"], eps, config["rope_parameters"]["rope_theta"], query=True, rounding=rounding)
        key = projection(normalized, positions, layer, config["num_key_value_heads"],
                         config["head_dim"], eps, config["rope_parameters"]["rope_theta"], rounding=rounding)
        value = rounding(normalized @ layer["v"].T).reshape(-1, config["num_key_value_heads"], config["head_dim"])
        attended = rounding(attention(query, key, value, positions, request_ids, histories[index], modes[index]))
        trace.append((query, key, value, attended))
        hidden = rounding(hidden + rounding(attended.reshape(len(hidden), -1) @ layer["o"].T))
        normalized = rms(hidden, layer["post_norm"], eps, rounding)
        gate = rounding(normalized @ layer["gate"].T)
        activated = rounding(gate / (1 + np.exp(-gate)))
        product = rounding(activated * rounding(normalized @ layer["up"].T))
        hidden = rounding(hidden + rounding(product @ layer["down"].T))
    return rms(hidden, parameters["final_norm"], eps, rounding), trace
