# Running on a GPU

The test suite runs anywhere, but the benchmarks need a CUDA GPU. This project was built on a machine
without one, so **these steps have not been run yet**. The kernels were only compiled for GPU targets
and run in the interpreter, so a first GPU run could expose a problem that neither of those can. If a
step fails, treat it as a real finding and fix the repo.

## Which GPU you get matters

What `tl.dot` compiles to depends on the architecture. This table comes from compiling
`_tiled_matmul_kernel` with blocks of 64 x 64 x 32 for each target, and reading the generated code:

| Target | Example GPUs | float32, `ieee` | float32, `tf32` | float16 |
|---|---|---|---|---|
| `sm_75` | T4, the usual free GPU | FMA loop | FMA loop | FMA loop |
| `sm_80`, `sm_86` | A100, A10 | FMA loop | MMA | MMA |
| `sm_90` | H100 | FMA loop | `warp_group_dot` (WGMMA) | `warp_group_dot` (WGMMA) |
| `sm_100` | B200 | FMA loop | `tc_gen5_mma` | `tc_gen5_mma` |

So a T4 can run every benchmark here, but it has no tensor cores and cannot show their effect. To see
MMA in the results, use an `sm_80` or newer GPU. A T4 also ignores `num_stages`, because the compiler
generates identical code for every value there, so differences across `num_stages` on a T4 are noise.

Check the GPU you were given against this table before you interpret the results. The example GPUs come
from general knowledge of each architecture and were not checked on hardware. Other architectures, such
as the `sm_89` of an L4, were not compiled here.

## Colab

1. Choose **Runtime > Change runtime type > GPU**, and create a new notebook.
2. Check that a GPU is attached:

   ```bash
   !nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
   ```

3. Clone and install. Colab already has a CUDA build of PyTorch, so pip keeps it if it satisfies the
   version range in `pyproject.toml`.

   ```bash
   !git clone https://github.com/<your-username>/TileKernels.git
   %cd TileKernels
   !pip install -e ".[dev]"
   ```

4. Run the tests compiled on the GPU. This is a correctness check on real hardware, and it should come
   before any timing. It includes behaviour the interpreter cannot show, such as `tf32` accuracy.

   ```bash
   !pytest --gpu
   ```

5. Run a small sweep first, to find any problems quickly. Then run the full sweeps.

   ```bash
   !python -m benchmarks.bench_softmax --quick --output-dir /tmp/quick
   !python -m benchmarks.bench_matmul --quick --output-dir /tmp/quick
   !python -m benchmarks.bench_softmax
   !python -m benchmarks.bench_matmul
   ```

   The matmul sweep compiles over a hundred configurations, so expect it to take a while. Its duration
   has not been measured.

6. Save the results. The simplest way that needs no credentials on Colab is to download them, then
   commit from your own machine.

   ```bash
   !zip -r results.zip results
   ```

   Download `results.zip` from the Files panel and unzip it over `results/` in your local clone.

## Kaggle

1. Create a notebook, then under **Settings > Accelerator** choose **GPU T4 x2**. Kernels and baselines
   use only the first GPU.
2. Follow steps 2 to 5 above, in a notebook cell.
3. Results are in `/kaggle/working/TileKernels/results/`. Zip them as above and download them from the
   Output panel.

## After the run

1. Read `results/SUMMARY.md`. It states which kernel won, using the numbers from your run.
2. Delete `results/README.md`, which says that results are pending.
3. Replace the pending section of the top-level `README.md` with the tables from `SUMMARY.md`, and link
   the plots.
4. Commit:

   ```bash
   git add results README.md
   git commit -m "docs: add measured GPU benchmark results"
   ```

## Optional: dump the stages from the real GPU

`scripts/dump_ir.py` works without a GPU, because it compiles for an explicit target. To dump for the GPU
you were given, pass its architecture, such as `--arch 75` for a T4. See
[COMPILER_DUMPS.md](COMPILER_DUMPS.md) for how to read the output, and for how to get disassembly.
