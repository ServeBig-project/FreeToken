"""Public contract: temperature <= 0 or top_k == 1 decodes greedily, whatever top_k/top_p say."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler
from freetoken.server.generation import resolve_sampling

QWEN = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}  # sampling defaults of a Qwen generation_config
GREEDY = [SamplingParams(0.0, 20, 0.95), SamplingParams(0.0, -1, 1.0), SamplingParams(0.0, 20, 0.5),
          SamplingParams(0.7, 1, 0.8), SamplingParams(1.5, 1, 1.0)]
SAMPLED = [SamplingParams(1.0, 20, 0.95), SamplingParams(0.7, -1, 1.0)]
ALL_GREEDY = GREEDY * 4
MIXED = [p for pair in zip(GREEDY * 4, SAMPLED * 10) for p in pair]  # greedy rows at even indices
VOCAB = 256


@pytest.mark.parametrize("top_p", [0.5, 0.8, 0.9, 0.95, 1.0])
@pytest.mark.parametrize("temperature,top_k,greedy", [
    (0.0, -1, True), (0.0, 20, True), (-1.0, 50, True), (0.7, 1, True), (1.5, 1, True),
    (0.7, 20, False), (1.0, -1, False), (1e-6, -1, False),
])
def test_is_greedy_depends_only_on_temperature_and_top_k(temperature, top_k, top_p, greedy):
    assert SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p).is_greedy is greedy


@pytest.mark.parametrize("model_sampling,request_fields,fields,greedy", [
    (QWEN, {"temperature": 0}, (0.0, 20, 0.95), True),
    (QWEN, {"temperature": 0, "top_p": 0.5}, (0.0, 20, 0.5), True),
    (QWEN, {}, (1.0, 20, 0.95), False),
    (QWEN, {"temperature": 0.8, "top_k": 1}, (0.8, 1, 0.95), True),
    (QWEN, {"temperature": 0.8}, (0.8, 20, 0.95), False),
    ({}, {}, (0.0, -1, 1.0), True),
])
def test_resolved_request_is_greedy_when_it_sets_temperature_zero(model_sampling, request_fields, fields, greedy):
    unset = {"temperature": None, "top_k": None, "top_p": None, "max_tokens": None, "ignore_eos": False}
    params = resolve_sampling(model_sampling=model_sampling, **{**unset, **request_fields})
    assert (params.temperature, params.top_k, params.top_p) == fields
    assert params.is_greedy is greedy


def _near_flat_logits(rows):
    # One clear winner over near-equal competitors: sampling at any real temperature often misses it.
    generator = torch.Generator().manual_seed(rows)
    logits = torch.rand(rows, VOCAB, generator=generator) * 0.01
    logits[torch.arange(rows), torch.randint(VOCAB, (rows,), generator=generator)] = 0.1
    return logits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Sampler.prepare pins host memory, needs CUDA")
@pytest.mark.parametrize("params", [ALL_GREEDY, MIXED], ids=["all-greedy", "mixed"])
def test_greedy_rows_sample_argmax_with_one_hot_probabilities(params):
    greedy_rows = [i for i, p in enumerate(params) if p in GREEDY]
    logits = _near_flat_logits(len(params)).cuda()
    expected = logits.argmax(-1)[greedy_rows]
    sampler = Sampler(torch.device("cuda"), VOCAB)
    args = sampler.prepare(SimpleNamespace(reqs=[SimpleNamespace(sampling_params=p) for p in params]))

    tokens = sampler.sample(logits.clone(), args)
    probs = sampler.probabilities(logits.clone(), args)

    assert tokens.shape == (len(params),) and probs.shape == (len(params), VOCAB)
    assert tokens[greedy_rows].tolist() == expected.tolist()
    assert torch.equal(probs[greedy_rows], torch.nn.functional.one_hot(expected, VOCAB).to(probs))
