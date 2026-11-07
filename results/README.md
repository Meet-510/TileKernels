# Results

**Pending: run on a GPU to fill this in.** No benchmark has been run yet, so this folder holds no
numbers. The machine this project was built on has no CUDA GPU, and timings from Triton's interpreter
mode are meaningless.

Follow [docs/RUNNING_ON_A_GPU.md](../docs/RUNNING_ON_A_GPU.md) to generate:

- `softmax.csv`, `matmul.csv`: every configuration, with the GPU and versions in a header
- `softmax_block_size.png`, `softmax_num_warps.png`, `matmul_block_size.png`, `matmul_schedule.png`
- `SUMMARY.md`: the best configurations, and an analysis that follows the measured numbers
