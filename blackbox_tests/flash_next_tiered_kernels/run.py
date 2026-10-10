"""Independent public-API numerical tests; no production source inspection."""

import argparse
import json
import math
from pathlib import Path

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention
from freetoken.kernel.triton.qsa.gather import gather_host_kv


PAGE, HKV, DIM, LAYERS = 64, 2, 256, 3
# Fixed before candidate execution; BF16 output rounds to roughly 0.8% precision.
ATOL, RTOL = 1 / 64, 1 / 64


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact(actual, expected, label):
    actual = actual.detach().cpu()
    require(actual.shape == expected.shape, f"{label}: shape mismatch")
    require(actual.dtype == expected.dtype, f"{label}: dtype mismatch")
    differences = actual != expected
    require(not differences.any().item(),
            f"{label}: {differences.sum().item()} unequal elements")


def guarded(shape, dtype, fill):
    count = math.prod(shape)
    storage = torch.full((count + 128,), fill, dtype=dtype, device="cuda")
    return storage, storage[64:64 + count].view(shape)


def patterns(kind, shift=0):
    if kind == "single":
        return [[shift]], [0], 1
    if kind == "maximum":
        return [list(range(2051))], [0], 33
    rows = [
        list(range(60, 64)) + [64, 65, 66],
        list(range(124, 128)) + [128, 129],
        [],
    ]
    if kind == "groups":
        rows += [
            list(range(0, 4)) + list(range(60, 64))
            + list(range(124, 128)) + list(range(188, 192)) + [252, 253, 254],
            list(range(4, 8)) + list(range(68, 72)) + [132],
        ]
    if shift:
        # Both phases consist of supported four-token groups and tail tokens.
        rows = [([x - 4 if x >= 4 else x + 4 for x in row]) for row in rows]
    return rows, [1, 0, 1, 0, 1][:len(rows)], 4


class Fixture:
    def __init__(self, dtype, kind, heads, mode, seed, layer, shift=0):
        self.dtype, self.kind, self.heads, self.layer = dtype, kind, heads, layer
        rows, requests, pages_per_request = patterns(kind, shift)
        self.tokens, self.width = len(rows), max(map(len, rows))
        self.indices = torch.full((self.tokens, self.width), -1, dtype=torch.int32)
        for i, row in enumerate(rows):
            self.indices[i, :len(row)] = torch.tensor(row, dtype=torch.int32)
        self.requests = torch.tensor(requests, dtype=torch.int32)
        self.pages = (max(requests) + 1) * pages_per_request
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(self.pages, generator=generator).to(torch.int32)
        self.tables = permutation.reshape(-1, pages_per_request)
        self.resident = torch.arange(self.pages).remainder(2).to(torch.int64)
        if mode != "mixed":
            self.resident.fill_(int(mode == "gpu"))
        if shift:
            self.resident = 1 - self.resident
        shape = (LAYERS, self.pages, 2, PAGE, HKV, DIM)
        if dtype == torch.int8:
            self.encoded = torch.randint(-80, 81, shape, generator=generator,
                                         dtype=torch.int8)
            self.encoded[..., ::7] = 0
            scales = torch.rand((LAYERS, self.pages, 2, PAGE, HKV),
                                generator=generator) * 0.027 + 0.003
            self.scales = scales.to(torch.bfloat16)
        else:
            self.encoded = (torch.randn(shape, generator=generator) * 0.7).to(dtype)
            self.scales = None
        self.host = alloc_pinned_tensor(*shape, dtype=dtype)
        self.host.copy_(self.encoded)
        self.host_scales = None
        fields = LAYERS
        if self.scales is not None:
            self.host_scales = alloc_pinned_tensor(*self.scales.shape,
                                                  dtype=torch.bfloat16)
            self.host_scales.copy_(self.scales)
            fields *= 2
        self.addresses = torch.empty((fields + 1, self.pages), dtype=torch.int64)
        for level in range(LAYERS):
            for page in range(self.pages):
                self.addresses[level, page] = device_ptr(self.host[level, page])
                if self.host_scales is not None:
                    self.addresses[LAYERS + level, page] = device_ptr(
                        self.host_scales[level, page])
        self.addresses[-1] = self.resident
        qshape = (self.tokens, heads, DIM)
        self.q = (torch.randn(qshape, generator=generator) * 0.4).to(torch.bfloat16)

    def full_cache(self):
        data = self.encoded[self.layer]
        scales = self.scales[self.layer] if self.scales is not None else None
        return data[:, 0].contiguous(), data[:, 1].contiguous(), scales

    def reference(self):
        keys, values, scales = self.full_cache()
        if scales is not None:
            keys = (keys.float() * scales[:, 0, :, :, None].float()).to(torch.bfloat16)
            values = (values.float() * scales[:, 1, :, :, None].float()).to(torch.bfloat16)
        result = torch.zeros_like(self.q)
        head_map = torch.arange(self.heads) // (self.heads // HKV)
        for row in range(self.tokens):
            selected = self.indices[row]
            selected = selected[selected >= 0].long()
            if not len(selected):
                continue
            pages = self.tables[self.requests[row], selected // PAGE].long()
            offsets = selected % PAGE
            k = keys[pages, offsets][:, head_map].double()
            v = values[pages, offsets][:, head_map].double()
            logits = torch.einsum("hd,shd->hs", self.q[row].double(), k) / math.sqrt(DIM)
            probabilities = torch.softmax(logits, dim=-1)
            result[row] = torch.einsum("hs,shd->hd", probabilities, v).to(torch.bfloat16)
        return result


class DeviceInputs:
    def __init__(self, fixture):
        self.q = fixture.q.cuda()
        self.indices = fixture.indices.cuda()
        self.tables = fixture.tables.cuda()
        self.requests = fixture.requests.cuda()
        self.addresses = torch.empty(tuple(reversed(fixture.addresses.shape)),
                                     dtype=torch.int64, device="cuda").T
        require(self.addresses.stride(0) == 1, "address table layout")
        require(self.addresses[-1].stride(0) > 1, "resident must exercise noncompact stride")
        keys, values, scales = fixture.full_cache()
        self.keys, self.values = keys.cuda(), values.cuda()
        self.ks = scales[:, 0].contiguous().cuda() if scales is not None else None
        self.vs = scales[:, 1].contiguous().cuda() if scales is not None else None
        shape = (fixture.tokens, fixture.width, HKV, DIM)
        self.storages, outputs = [], []
        fills = [-117, 109, -3, 5] if fixture.dtype == torch.int8 else [-19.5, 11.25]
        for number, fill in enumerate(fills):
            dtype = fixture.dtype if number < 2 else torch.bfloat16
            storage, output = guarded(shape if number < 2 else shape[:-1], dtype, fill)
            self.storages.append(storage)
            outputs.append(output)
        self.fills = fills
        self.outputs = tuple(outputs + [None] * (4 - len(outputs)))
        self.out_storage, self.out = guarded(tuple(fixture.q.shape), torch.bfloat16, 17)
        self.load(fixture)

    def load(self, fixture):
        self.q.copy_(fixture.q)
        self.indices.copy_(fixture.indices)
        self.tables.copy_(fixture.tables)
        self.requests.copy_(fixture.requests)
        self.addresses.copy_(fixture.addresses)
        keys, values, scales = fixture.full_cache()
        keys, values = keys.clone(), values.clone()
        host = fixture.resident == 0
        # Poison absent GPU payload so a wrong residency decision is observable.
        keys[host] = 71
        values[host] = -67
        self.keys.copy_(keys)
        self.values.copy_(values)
        if scales is not None:
            ks, vs = scales[:, 0].clone(), scales[:, 1].clone()
            ks[host], vs[host] = 3, 5
            self.ks.copy_(ks)
            self.vs.copy_(vs)
        self.reset_outputs()

    def reset_outputs(self):
        for storage, fill in zip(self.storages, self.fills):
            storage.fill_(fill)
        self.out_storage.fill_(17)

    def gather(self, fixture):
        gather_host_kv(self.indices, self.tables, self.requests, self.addresses,
                       self.outputs, layer=fixture.layer, layers=LAYERS, page_size=PAGE)

    def attend(self, supplied_out=True):
        return qsa_sparse_paged_attention(
            self.q, self.keys, self.values, self.indices, self.tables, self.requests,
            out=self.out if supplied_out else None, k_scale=self.ks, v_scale=self.vs,
            gathered=self.outputs, resident=self.addresses[-1])

    def check_gather(self, fixture):
        expected = [torch.full(tuple(output.shape), fill, dtype=output.dtype)
                    for output, fill in zip(self.outputs, self.fills)]
        for row in range(fixture.tokens):
            for slot, token in enumerate(fixture.indices[row].tolist()):
                if token < 0:
                    continue
                page = int(fixture.tables[fixture.requests[row], token // PAGE])
                if fixture.resident[page]:
                    continue
                for payload in range(2):
                    expected[payload][row, slot] = fixture.encoded[
                        fixture.layer, page, payload, token % PAGE]
                    if fixture.scales is not None:
                        expected[2 + payload][row, slot] = fixture.scales[
                            fixture.layer, page, payload, token % PAGE]
        for number, (storage, values, fill) in enumerate(zip(self.storages, expected, self.fills)):
            with_guards = torch.full(tuple(storage.shape), fill, dtype=storage.dtype)
            with_guards[64:-64] = values.flatten()
            exact(storage, with_guards, f"gather payload {number}, guards/GPU/padding included")

    def check_attention(self, fixture, actual):
        require(isinstance(actual, torch.Tensor), "attention must return a tensor without out")
        keys, values, scales = fixture.full_cache()
        baseline = qsa_sparse_paged_attention(
            self.q, keys.cuda(), values.cuda(), self.indices, self.tables, self.requests,
            k_scale=scales[:, 0].contiguous().cuda() if scales is not None else None,
            v_scale=scales[:, 1].contiguous().cuda() if scales is not None else None)
        exact(actual, baseline.cpu(), "tiered versus all-GPU")
        reference = fixture.reference()
        actual_cpu = actual.detach().cpu()
        require(actual_cpu.dtype == torch.bfloat16, "attention output dtype")
        require(actual_cpu.shape == reference.shape, "attention output shape")
        close = torch.isclose(actual_cpu.float(), reference.float(), atol=ATOL, rtol=RTOL)
        max_error = (actual_cpu.float() - reference.float()).abs().max().item()
        require(close.all().item(),
                f"independent softmax reference: {(~close).sum().item()} outside tolerance; "
                f"maximum absolute error={max_error}")
        empty = (fixture.indices == -1).all(dim=1)
        exact(actual_cpu[empty], torch.zeros_like(actual_cpu[empty]), "all-padding attention")
        guards = torch.cat((self.out_storage[:64], self.out_storage[-64:]))
        exact(guards, torch.full((128,), 17, dtype=torch.bfloat16), "attention output guards")


def eager_case(dtype, kind, heads, mode, layer):
    fixture = Fixture(dtype, kind, heads, mode, seed=341, layer=layer)
    device = DeviceInputs(fixture)
    device.gather(fixture)
    torch.cuda.synchronize()
    device.check_gather(fixture)
    returned = device.attend(supplied_out=False)
    device.check_attention(fixture, returned)
    device.attend(supplied_out=True)
    device.check_attention(fixture, device.out)


def graph_case(dtype):
    fixtures = [Fixture(dtype, "groups", 4, "mixed", seed, layer=1, shift=shift)
                for seed, shift in [(341, 0), (982, 1)]]
    device = DeviceInputs(fixtures[0])
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        device.gather(fixtures[0])
        device.attend()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        device.gather(fixtures[0])
        device.attend()
    for iteration in [0, 1, 0, 1]:
        fixture = fixtures[iteration]
        device.load(fixture)
        graph.replay()
        torch.cuda.synchronize()
        device.check_gather(fixture)
        device.check_attention(fixture, device.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["all", "eager", "graph"], default="all")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    cases = []
    if args.suite in ("all", "eager"):
        for dtype, kind, heads, mode, layer in [
            (torch.bfloat16, "single", 2, "host", 0),
            (torch.bfloat16, "boundary", 4, "mixed", 1),
            (torch.bfloat16, "groups", 8, "mixed", 2),
            (torch.int8, "groups", 8, "mixed", 2),
            (torch.int8, "single", 2, "host", 0),
            (torch.int8, "boundary", 4, "gpu", 1),
            (torch.bfloat16, "groups", 4, "host", 0),
            (torch.bfloat16, "maximum", 4, "mixed", 1),
            (torch.int8, "maximum", 4, "mixed", 1),
        ]:
            name = f"eager/{dtype}/{kind}/heads={heads}/{mode}/layer={layer}"
            cases.append((name, lambda d=dtype, k=kind, h=heads, m=mode, l=layer:
                          eager_case(d, k, h, m, l)))
    if args.suite in ("all", "graph"):
        for dtype in [torch.bfloat16, torch.int8]:
            cases.append((f"graph/{dtype}/address-residency-selection-replay",
                          lambda d=dtype: graph_case(d)))
    results = []
    for name, run in cases:
        try:
            run()
            result = {"name": name, "status": "passed"}
        except Exception as error:
            result = {"name": name, "status": "failed", "error_type": type(error).__name__,
                      "error": str(error).strip().splitlines()[0] if str(error).strip() else ""}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report = {"reference_atol": ATOL, "reference_rtol": RTOL, "results": results}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
