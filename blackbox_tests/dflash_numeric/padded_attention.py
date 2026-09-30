"""Callback with padded PyTorch attention, independent of the scalar NumPy oracle."""

import numpy as np
import torch


def attend(queries, keys, values, positions, request_ids, histories, mode):
    requests = np.unique(request_ids)
    query_counts = [int(np.sum(request_ids == request_id)) for request_id in requests]
    key_counts = [len(histories[request_id][0]) + count for request_id, count in zip(requests, query_counts)]
    batch, heads, width = len(requests), queries.shape[1], queries.shape[-1]
    qpad = torch.full((batch, max(query_counts), heads, width), 9.0, dtype=torch.float32, device=queries.device)
    kpad = torch.full((batch, max(key_counts), keys.shape[1], width), 7.0, dtype=torch.float32, device=queries.device)
    vpad = torch.full_like(kpad, -5.0)
    qpos = torch.zeros((batch, max(query_counts)), dtype=torch.long, device=queries.device)
    kpos = torch.zeros((batch, max(key_counts)), dtype=torch.long, device=queries.device)
    valid_keys = torch.zeros((batch, max(key_counts)), dtype=torch.bool, device=queries.device)
    rows = []
    for batch_index, request_id in enumerate(requests):
        selected = np.flatnonzero(request_ids == request_id)
        rows.append(selected)
        old_positions, old_keys, old_values = histories[request_id]
        count, total = query_counts[batch_index], key_counts[batch_index]
        qpad[batch_index, :count] = queries[selected].float()
        kpad[batch_index, :total] = torch.cat((old_keys, keys[selected])).float()
        vpad[batch_index, :total] = torch.cat((old_values, values[selected])).float()
        selected_positions = torch.tensor(positions[selected], device=queries.device)
        qpos[batch_index, :count] = selected_positions
        kpos[batch_index, :total] = torch.cat((old_positions, selected_positions))
        valid_keys[batch_index, :total] = True
    repeated_keys = kpad.repeat_interleave(heads // keys.shape[1], dim=2)
    repeated_values = vpad.repeat_interleave(heads // keys.shape[1], dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", qpad, repeated_keys) / np.sqrt(width)
    visible = valid_keys[:, None, :].expand(-1, qpad.shape[1], -1)
    causal, window_left = mode
    if causal:
        visible = visible & (kpos[:, None, :] <= qpos[:, :, None])
    if window_left >= 0:
        visible = visible & (kpos[:, None, :] >= qpos[:, :, None] - window_left)
    scores.masked_fill_(~visible[:, None, :, :], -torch.inf)
    probability = torch.softmax(scores, dim=-1)
    padded_output = torch.einsum("bhqk,bkhd->bqhd", probability, repeated_values)
    output = torch.empty_like(queries)
    for batch_index, selected in enumerate(rows):
        output[selected] = padded_output[batch_index, :len(selected)].to(queries.dtype)
    return output
