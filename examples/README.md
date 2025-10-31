The examples can run on both Nvidia GPU (using NCCL backend) and CPU (using GLOO backend).  

```bash
uv run torchrun --nproc_per_node=3 examples/row_parallel_linear_demo.py
```

```bash
uv run torchrun --nproc_per_node=4 examples/row_parallel_linear_demo.py
```