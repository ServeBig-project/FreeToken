"""P0 operator-level black-box suite for the shared runtime pool (contract section 8).

Run with the implementation on PYTHONPATH and one GPU visible, e.g.
    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=<impl>/python python -m pytest blackbox_tests/runtime_pool

Every layout test runs the same public operator on a compact per-layer pool and on the
production slot-major pool (storage [slots, layers, ...] viewed as [layers, slots, ...],
then one layer selected), with identical contents, and requires bit-identical results.
Math references are a sanity check only; the layout comparison is the acceptance check.
"""

import torch
import torch.nn.functional as F

DEV = "cuda"
BF16 = torch.bfloat16
HK, HV, K, V = 16, 32, 128, 128  # Qwen3.6-35B-A3B GDN layer, TP=1
KW = 4
D = 2 * HK * K + HV * V  # 8192
REAL_LAYERS = 30
SCALE = K ** -0.5
G = HV // HK


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(g, *shape, dtype=torch.float32, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).to(DEV, dtype)


def ints(values, dtype=torch.int32):
    return torch.tensor(values, device=DEV, dtype=dtype)


class Pool:
    """A slot-major storage [n, layers, ...] and a compact layer-major copy [layers, n, ...]
    with identical contents. `strided`/`compact` are the per-layer views of `layer`."""

    def __init__(self, g, n, layers, layer, inner, dtype, scale=1.0):
        self.storage = randn(g, n, layers, *inner, dtype=dtype, scale=scale)
        self.layer = layer
        self.strided_lm = self.storage.transpose(0, 1)  # [layers, n, ...], row stride = layers x inner
        self.compact_lm = self.strided_lm.contiguous()
        self.strided = self.strided_lm[layer]
        self.compact = self.compact_lm[layer]
        self.snapshot = self.storage.clone()
        assert self.strided.stride(0) == layers * self.compact.stride(0)

    def assert_same(self, rows, layers=None):
        rows = list(rows)
        for l in [self.layer] if layers is None else layers:
            assert torch.equal(self.strided_lm[l][rows], self.compact_lm[l][rows])

    def assert_untouched(self, rows, layers=None):
        """Everything outside [rows] x [layers] (default: this pool's layer) is unchanged,
        in both the slot-major storage and the compact copy."""
        layers = [self.layer] if layers is None else list(layers)
        keep = torch.ones(self.storage.shape[:2], dtype=torch.bool, device=DEV)
        for r in rows:
            keep[r, layers] = False
        assert torch.equal(self.storage[keep], self.snapshot[keep])
        assert torch.equal(self.compact_lm.transpose(0, 1)[keep], self.snapshot[keep])


def gates(a, b, A_log, dt_bias):
    g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    return g, torch.sigmoid(b.float())


def gdn_ref(S, q, k, v, g, beta, scale=SCALE):
    """Gated delta rule, fp32. S [HV, V, K] (y = S @ q); q/k [T, HK, K]; v [T, HV, V];
    g/beta [T, HV]. Returns (o [T, HV, V], final S, u [T, HV, V], normalized k [T, HK, K])."""
    S = S.float().clone()
    outs, us, ks = [], [], []
    for t in range(q.shape[0]):
        qn = F.normalize(q[t].float(), dim=-1)
        kn = F.normalize(k[t].float(), dim=-1)
        qq, kk = qn.repeat_interleave(G, 0), kn.repeat_interleave(G, 0)
        S = S * torch.exp(g[t].float())[:, None, None]
        u = beta[t].float()[:, None] * (v[t].float() - torch.einsum("hvk,hk->hv", S, kk))
        S = S + u[:, :, None] * kk[:, None, :]
        outs.append(torch.einsum("hvk,hk->hv", S, qq) * scale)
        us.append(u)
        ks.append(kn)
    return torch.stack(outs), S, torch.stack(us), torch.stack(ks)


def conv_ref(history, x, weight):
    """Depthwise causal conv + silu, fp32. history [KW-1, D] (oldest first); x [T, D]."""
    full = torch.cat([history.float(), x.float()])
    w = weight.float().T  # [KW, D]
    return torch.stack([F.silu((full[t:t + KW] * w).sum(0)) for t in range(x.shape[0])])


def assert_close(actual, expected, atol, rtol):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
