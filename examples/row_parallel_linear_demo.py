import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn.functional as dnnF
from torch.nn import functional as F


def init_dist():
    """Initialize distributed backend and return (rank, world, device)."""
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world = dist.get_world_size()

    # Assign each rank its device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", rank))  # fallback to rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world, device


class RowParallelLinear(nn.Module):
    """
    Row-sharded Linear:
      - Full weight: [in_features, out_features]
      - Each rank holds shard: [in_features/world_size, out_features]
      - Forward:
          x_local = x[:, in_slice]
          y_local = x_local @ W_local
          y = SUM_r(y_local) via all_reduce
      - Backward:
          Autograd computes local grads for W_local and x_local.
          No input hook needed since x is already partitioned.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert in_features % world == 0, "in_features must be divisible by world size"

        self.world = world
        self.rank = rank
        self.in_per_rank = in_features // world
        self.out_features = out_features

        # Local row shard of the weight
        self.weight = nn.Parameter(torch.empty(self.in_per_rank, out_features))
        # Full bias (identical across ranks)
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Slice for local input chunk
        start = rank * self.in_per_rank
        end = start + self.in_per_rank
        self._in_slice = slice(start, end)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in_full = self.weight.size(0) * self.world
            bound = 1.0 / (fan_in_full ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local input chunk
        x_local = x[:, self._in_slice]

        # Local matmul → partial output [B, Dout]
        y_local = x_local @ self.weight

        # Sum partial outputs across ranks → full y
        if self.world > 1:
            y_local = dnnF.all_reduce(y_local)  # autograd-safe

        # Bias added identically on all ranks
        if self.bias is not None:
            y_local = y_local + self.bias

        return y_local


def main():
    rank, world, device = init_dist()
    torch.manual_seed(0 + rank)

    B, Din, Dout = 4, 8, 12
    assert Din % world == 0

    layer = RowParallelLinear(Din, Dout, bias=True).to(device)

    # Toy input/target
    x = torch.randn(B, Din, device=device, requires_grad=True)
    t = torch.randint(0, Dout, (B,), device=device)

    # Forward + Backward
    y = layer(x)
    loss = F.cross_entropy(y, t)
    loss.backward()

    if rank == 0:
        print("RowParallelLinear OK:",
              dict(device=str(device),
                   y=y.shape,
                   x_grad=(x.grad is not None),
                   w_grad=(layer.weight.grad is not None),
                   b_grad=(layer.bias.grad is not None if layer.bias is not None else None)))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()