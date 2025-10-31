import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn.functional as dnnF
from torch.nn import functional as F


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


class RowParallelLinear(nn.Module):
    """
    Row-sharded Linear:
      - Full weight: [in_features, out_features]
      - Each rank holds shard: [in_features/world_size, out_features]  (rows of W)
      - Forward:
          x is sliced on features: x_local = x[:, in_slice]
          y_local = x_local @ W_local
          y = SUM_r(y_local)  via all_reduce
      - Backward:
          autograd gives local grads for W_local and x_local.
          No special hook needed for x here because we explicitly sliced x (each rank
          owns its chunk). The output sum is correct via all_reduce in forward.
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

        # local shard of rows
        self.weight = nn.Parameter(torch.empty(self.in_per_rank, out_features))
        # Bias is logically full-sized; keep one per rank (identical updates in lockstep).
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

        # Precompute our input slice
        start = rank * self.in_per_rank
        end = start + self.in_per_rank
        self._in_slice = slice(start, end)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(0) * self.world  # effective fan_in of full W
            bound = 1.0 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scatter input features to this rank
        # x_full: [B, Din] -> x_local: [B, Din/world]
        x_local = x[:, self._in_slice]

        # Local matmul → partial output [B, Dout]
        y_local = x_local.matmul(self.weight)

        # Sum partial outputs across ranks to get full y
        if self.world > 1:
            y_local = dnnF.all_reduce(y_local)

        # Bias is full-sized; add on every rank (grads identical across ranks)
        if self.bias is not None:
            y_local = y_local + self.bias

        return y_local


def main():
    rank, world = init_dist()
    torch.manual_seed(0)

    B, Din, Dout = 4, 8, 12
    assert Din % world == 0

    layer = RowParallelLinear(Din, Dout, bias=True)
    # Toy input/target
    x = torch.randn(B, Din, requires_grad=True)
    t = torch.randint(0, Dout, (B,))

    # Forward
    y = layer(x)                 # [B, Dout]
    loss = F.cross_entropy(y, t)

    # Backward
    loss.backward()

    if rank == 0:
        print("RowParallelLinear OK:",
              dict(y=y.shape,
                   x_grad=(x.grad is not None),
                   w_grad=(layer.weight.grad is not None),
                   b_grad=(layer.bias.grad is not None if layer.bias is not None else None)))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()