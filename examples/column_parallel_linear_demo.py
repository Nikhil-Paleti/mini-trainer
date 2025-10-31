import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn.functional as dnnF
from torch.nn import functional as F


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


class ColumnParallelLinear(nn.Module):
    """
    Column-sharded Linear:
      - Full weight: [in_features, out_features]
      - Each rank holds shard: [in_features, out_features/world_size]
      - Forward: y_local = x @ W_local (+ b_local)
      - If gather_output=True → y = all_gather(y_local) then concat on last dim
      - Backward: autograd handles param grads; we all_reduce input grad so it equals dense Linear's grad.
    """
    def __init__(self, in_features, out_features, bias=True, gather_output=True):
        super().__init__()
        assert out_features % dist.get_world_size() == 0
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.in_features = in_features
        self.out_per_rank = out_features // self.world
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(in_features, self.out_per_rank))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming-uniform similar to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(0)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Register a hook to all-reduce grad wrt input so it matches dense Linear
        def _allreduce_input_grad(g: torch.Tensor) -> torch.Tensor:
            if dist.is_available() and dist.is_initialized() and self.world > 1:
                dist.all_reduce(g, op=dist.ReduceOp.SUM)
            return g

        if x.requires_grad:
            x.register_hook(_allreduce_input_grad)

        y_local = x.matmul(self.weight)  # [B, out_per_rank]
        if self.bias is not None:
            y_local = y_local + self.bias

        if self.gather_output and self.world > 1:
            # Autograd-aware gather (backward splits grad correctly to each shard)
            parts = dnnF.all_gather(y_local)  # list of [B, out_per_rank]
            y = torch.cat(parts, dim=-1)      # [B, out_features]
            return y
        else:
            return y_local


def main():
    rank, world = init_dist()
    torch.manual_seed(0)

    B, Din, Dout = 4, 8, 12
    assert Dout % world == 0

    layer = ColumnParallelLinear(Din, Dout, bias=True, gather_output=True)
    # Toy input/target
    x = torch.randn(B, Din, requires_grad=True)
    t = torch.randint(0, Dout, (B,))

    # Forward
    y = layer(x)                 # [B, Dout]
    loss = F.cross_entropy(y, t)

    # Backward
    loss.backward()

    if rank == 0:
        print("ColumnParallelLinear OK:",
              dict(y=y.shape, x_grad=(x.grad is not None),
                   w_grad=(layer.weight.grad is not None),
                   b_grad=(layer.bias.grad is not None if layer.bias is not None else None)))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()