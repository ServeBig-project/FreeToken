"""Fetch only selected host K/V cells into per-query device scratch, without host selection."""
import triton
import triton.language as tl


@triton.jit
def _gather(indices, tables, requests, addresses, out_k, out_v, out_ks, out_vs, counter,
            index_stride, table_stride, address_stride,
            TABLE_WIDTH: tl.constexpr, REQUESTS: tl.constexpr, PAGES: tl.constexpr,
            WIDTH: tl.constexpr, PAGE_SIZE: tl.constexpr, HEADS: tl.constexpr,
            DIM: tl.constexpr, LAYER: tl.constexpr, LAYERS: tl.constexpr,
            FAMILIES: tl.constexpr, INT8: tl.constexpr, BLOCK_D: tl.constexpr,
            BLOCK_H: tl.constexpr, COUNT: tl.constexpr, ELEMENT_BYTES: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * 4 + tl.arange(0, 4)
    req = tl.load(requests + row)
    token = tl.load(indices + row * index_stride + columns, columns < WIDTH, other=-1)
    page = tl.maximum(token, 0) // PAGE_SIZE
    valid = (columns < WIDTH) & (token >= 0) & (page < TABLE_WIDTH) & (req >= 0) & (req < REQUESTS)
    physical = tl.load(tables + tl.maximum(req, 0).to(tl.int64) * table_stride + page,
                       valid, other=0).to(tl.int64)
    valid &= (physical >= 0) & (physical < PAGES)
    on_gpu = tl.load(addresses + physical * address_stride + FAMILIES, valid, other=1)
    valid &= on_gpu == 0
    base = tl.load(addresses + physical * address_stride + LAYER, valid, other=0)
    features = tl.arange(0, BLOCK_D)
    offset = (token % PAGE_SIZE)[:, None] * HEADS * DIM + features[None, :]
    dest = ((row * WIDTH + columns[:, None]) * HEADS * DIM + features[None, :])
    mask = valid[:, None] & (features[None, :] < HEADS * DIM)
    source = base.to(tl.pointer_type(out_k.dtype.element_ty))
    k = tl.load(source[:, None] + offset, mask, other=0)
    v = tl.load(source[:, None] + PAGE_SIZE * HEADS * DIM + offset, mask, other=0)
    tl.store(out_k + dest, k, mask)
    tl.store(out_v + dest, v, mask)
    if INT8:
        heads = tl.arange(0, BLOCK_H)
        scale_base = tl.load(addresses + physical * address_stride + LAYERS + LAYER,
                             valid, other=0).to(tl.pointer_type(out_ks.dtype.element_ty))
        offset_s = (token % PAGE_SIZE)[:, None] * HEADS + heads[None, :]
        dest_s = (row * WIDTH + columns[:, None]) * HEADS + heads[None, :]
        scale_mask = valid[:, None] & (heads[None, :] < HEADS)
        ks = tl.load(scale_base[:, None] + offset_s, scale_mask, other=0)
        vs = tl.load(scale_base[:, None] + PAGE_SIZE * HEADS + offset_s, scale_mask, other=0)
        tl.store(out_ks + dest_s, ks, scale_mask)
        tl.store(out_vs + dest_s, vs, scale_mask)
    if COUNT:
        cells = tl.sum(valid.to(tl.int32))
        if cells > 0:
            tl.atomic_add(counter, cells * HEADS * (2 * DIM * ELEMENT_BYTES + (4 if INT8 else 0)),
                          sem="relaxed")


def gather_host_kv(indices, tables, requests, addresses, outputs, *, layer, layers, page_size,
                   counter=None):
    k, v, ks, vs = outputs
    heads, dim = k.shape[-2:]
    _gather[(indices.shape[0], triton.cdiv(indices.shape[1], 4))](
        indices, tables, requests, addresses, k, v, ks, vs, counter,
        indices.stride(0), tables.stride(0), addresses.stride(1),
        TABLE_WIDTH=tables.shape[1], REQUESTS=tables.shape[0], PAGES=addresses.shape[1],
        WIDTH=indices.shape[1], PAGE_SIZE=page_size, HEADS=heads, DIM=dim,
        LAYER=layer, LAYERS=layers, FAMILIES=addresses.shape[0] - 1,
        INT8=ks is not None, BLOCK_D=triton.next_power_of_2(heads * dim),
        BLOCK_H=triton.next_power_of_2(heads), COUNT=counter is not None,
        ELEMENT_BYTES=k.element_size(), num_warps=4)
