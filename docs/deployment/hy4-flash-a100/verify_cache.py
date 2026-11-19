"""A/B the per-step cached prepare_groups against the previous implementation."""
import time
import torch
from vllm.models.hy_v4_flash.nvidia import indexpool_prepare_torch as new
from vllm.models.hy_v4_flash.nvidia import indexpool_prepare_ref as ref

dev = torch.device("cuda")
torch.manual_seed(0)
RATIO, RING, DIM, BLOCKS = 4, 64, 128, 8192


def make(tokens, batch, ndim3=True):
    keys = torch.randn(tokens, DIM, dtype=torch.bfloat16, device=dev) * 0.05
    positions = (torch.arange(tokens, device=dev).repeat(3, 1)
                 if ndim3 else torch.arange(tokens, device=dev))
    tail = torch.zeros(BLOCKS, RING, 131, dtype=torch.float32, device=dev)
    table = torch.arange(batch, device=dev, dtype=torch.int64).repeat(1, 64)
    seg = max(tokens // batch, 1)
    starts = torch.arange(0, batch + 1, device=dev, dtype=torch.int32) * seg
    lengths = (starts[1:] - starts[:-1]).clone()
    return keys, positions, tail, table, starts, lengths


def cmp(tag, a, b):
    assert len(a) == len(b), tag
    worst = 0.0
    for x, y in zip(a, b):
        assert x.shape == y.shape, f"{tag}: {tuple(x.shape)} vs {tuple(y.shape)}"
        if x.dtype.is_floating_point:
            worst = max(worst, (x.float() - y.float()).abs().max().item())
        else:
            assert torch.equal(x.to(torch.int64), y.to(torch.int64)), f"{tag}: int mismatch"
    print(f"  {tag:26s} max|d|={worst:.3e} {'OK' if worst < 1e-3 else 'MISMATCH'}")


for tokens, batch, ndim3 in ((1, 1, True), (6, 2, True), (2048, 4, True), (8192, 1, False)):
    args = make(tokens, batch, ndim3)
    cmp(f"tokens={tokens} batch={batch}", ref.prepare_groups(*args, tokens, RATIO),
        new.prepare_groups(*args, tokens, RATIO))
    # cache hit path: same tensors again
    cmp(f"  (cache hit) tokens={tokens}", ref.prepare_groups(*args, tokens, RATIO),
        new.prepare_groups(*args, tokens, RATIO))


def bench(label, fn, n=20):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f"  {label:26s} {(time.time()-t0)/n*1000:6.2f} ms")


for tokens, batch, tag in ((1, 1, "decode 1 tok"), (6, 2, "decode k=6"), (8192, 1, "prefill 8K")):
    args = make(tokens, batch, True)
    print(f"== {tag}")
    bench("previous impl", lambda: ref.prepare_groups(*args, tokens, RATIO))
    new._STEP_CACHE["key"] = None  # force a miss each call: worst case
    bench("cached (miss)", lambda: new.prepare_groups(*args, tokens, RATIO))
    new._STEP_CACHE["key"] = None
    new.prepare_groups(*args, tokens, RATIO)
    bench("cached (hit)", lambda: new.prepare_groups(*args, tokens, RATIO))
