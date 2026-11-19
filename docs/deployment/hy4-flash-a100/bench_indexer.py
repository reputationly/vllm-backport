"""Cost of the Hy4 IndexPool path at real decode/prefill shapes (single GPU)."""
import torch, time
from vllm.models.hy_v4_flash.nvidia.indexpool_prepare_torch import prepare_groups
from vllm.v1.attention.ops.triton_mla_sparse_kernel import *  # noqa: F401

dev = torch.device("cuda")
torch.manual_seed(0)
RATIO, RING, DIM, HEADS, GROUP_TOPK = 4, 64, 128, 16, 2048
MAXLEN = 16384
BLOCKS = 4096


def bench(label, fn, n=10):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f"  {label:44s} {(time.time()-t0)/n*1000:7.2f} ms", flush=True)


def run(tokens, tag):
    keys = (torch.randn(tokens, DIM, dtype=torch.bfloat16, device=dev) * 0.05)
    positions = torch.arange(tokens, dtype=torch.int64, device=dev).repeat(3, 1)
    tail = torch.zeros(BLOCKS, RING, 131, dtype=torch.float32, device=dev)
    table = torch.arange(1, dtype=torch.int64, device=dev).repeat(1, 64)
    starts = torch.tensor([0, tokens], dtype=torch.int32, device=dev)
    lengths = torch.tensor([tokens], dtype=torch.int32, device=dev)
    print(f"== {tag}: tokens={tokens}")
    bench("prepare_groups (torch)", lambda: prepare_groups(
        keys, positions, tail, table, starts, lengths, tokens, RATIO))

    # logits + topk stand-ins at the same row count
    q = torch.randn(tokens, 1, 32, DIM, dtype=torch.bfloat16, device=dev)
    logits = torch.randn(tokens, MAXLEN // RATIO, dtype=torch.float32, device=dev)
    bench(f"topk(logits) group_topk={GROUP_TOPK}", lambda: torch.topk(logits, GROUP_TOPK, dim=-1))


run(1, "decode")
run(6, "decode k=6")
run(2048, "prefill 2K")
run(8192, "prefill 8K")
