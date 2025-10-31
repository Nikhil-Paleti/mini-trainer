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


class ColumnParallelLinear(nn.Module):
    """
    Column-sharded Linear:
      - Full weight: [in_features, out_features]
      - Each rank holds shard: [in_features, out_features/world_size]
      - Forward:
          y_local = x @ W_local (+ b_local)
          if gather_output → concat(all_gather(y_local)) along last dim
      - Backward:
          Autograd handles local param grads.
          Register hook to all_reduce input grads for DDP-style correctness.
    """
    def __init__(self, in_features, out_features, bias=True, gather_output=True):
        super().__init__()
        world = dist.get_world_size()
        rank = dist.get_rank()
        assert out_features % world == 0, "out_features must be divisible by world size"

        self.world = world
        self.rank = rank
        self.in_features = in_features
        self.out_per_rank = out_features // world
        self.gather_output = gather_output

        # Local column shard
        self.weight = nn.Parameter(torch.empty(in_features, self.out_per_rank))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(0)
            bound = 1.0 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hook to all-reduce input grad → ensures identical gradients across ranks
        def _allreduce_input_grad(grad):
            if dist.is_initialized() and self.world > 1:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            return grad

        if x.requires_grad:
            x.register_hook(_allreduce_input_grad)

        # Local matmul: partial output [B, out_per_rank]
        y_local = x @ self.weight

        if self.bias is not None:
            y_local = y_local + self.bias

        if self.gather_output and self.world > 1:
            # Autograd-aware gather (differentiable)
            parts = dnnF.all_gather(y_local)
            y = torch.cat(parts, dim=-1)  # [B, out_features]
            return y
        else:
            return y_local


def main():
    rank, world, device = init_dist()
    torch.manual_seed(0 + rank)

    B, Din, Dout = 4, 8, 12
    assert Dout % world == 0

    layer = ColumnParallelLinear(Din, Dout, bias=True, gather_output=True).to(device)

    # Toy input/target
    x = torch.randn(B, Din, device=device, requires_grad=True)
    t = torch.randint(0, Dout, (B,), device=device)

    # Forward + Backward
    y = layer(x)
    loss = F.cross_entropy(y, t)
    loss.backward()

    if rank == 0:
        print("ColumnParallelLinear OK:",
              dict(device=str(device),
                   y=y.shape,
                   x_grad=(x.grad is not None),
                   w_grad=(layer.weight.grad is not None),
                   b_grad=(layer.bias.grad is not None if layer.bias is not None else None)))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()