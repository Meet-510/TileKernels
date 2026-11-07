# Kernel notes

The math and memory behaviour of each kernel, and the Triton concepts they rely on. Numbers about generated
code come from [dumps/experiments/results.md](../dumps/experiments/results.md). They describe the code and
are not timings.

## Triton concepts

**Program.** One instance of a kernel, launched per element of the grid. A program runs on a CUDA block, and
`tl.program_id` says which one it is. The grids here are 1D: one program per row, or per tile.

**Block.** A value in a kernel is a tensor whose size is known at compile time. Triton decides which thread
holds which element. Block sizes must be powers of two, because `tl.arange` requires it.

**`num_warps`.** How many warps of 32 threads run each program. More threads can process a block in
parallel, but crossing warps costs shared memory and barriers for some operations.

**`num_stages`.** How many stages the software pipeliner uses for a loop on `sm_80` and newer. A value above
1 starts fetching the next tile's data while the current one is being used, so memory latency overlaps with
computation, at the cost of more shared memory.

**constexpr.** An argument fixed at compile time. A new value means a new compilation, which is why block
sizes must be constexprs: they determine tensor types.

## Why masks are needed

A block's size is fixed at compile time, but the data's shape is only known at run time, and is rarely an
exact multiple. Reading or writing a whole block would touch memory past the end of the tensor, or data from
the next row. A mask makes each load and store conditional for each element.

The alternative is padding the data up to a multiple of the block. That means copying data and using more
memory, while a mask costs a comparison per element: 14% more code in the softmax experiment.

What a masked element takes as its value is part of each kernel's design:

| Kernel | `other` | Why it is right |
|---|---|---|
| Softmax | `-inf` | `exp(-inf - max) = 0`, so it adds nothing to the sum and cannot be the max |
| Matmul | `0` | A row or column of zeros adds nothing to a matrix product |

Note that `0` would be wrong for softmax: it would add `exp(0 - max)` to the denominator, and could even
become the max. `tests/test_mask_mutations.py` shows this failing.

Stores must be masked as well. A missing store mask leaves the in-shape values correct and overwrites whatever
is next to them. Comparing outputs alone cannot catch this, so the stride tests check the gaps around an output.

## Fused softmax

For one row `x` with `n` columns:

```text
y_i = exp(x_i) / sum_j exp(x_j)
```

**Stability.** `exp(x)` overflows float32 above about 88. Dividing the top and bottom by `exp(m)`, where
`m = max_j x_j`, gives the same result with safe exponents:

```text
y_i = exp(x_i - m) / sum_j exp(x_j - m)
```

Now the largest exponent is exactly 0, so no term exceeds 1, and the denominator is between 1 and `n`.

**Memory access.** One program per row. Each reads `n` elements once and writes `n` elements once, so the whole
kernel moves `2 * n_rows * n_cols` elements. The PyTorch version, if each step is a separate kernel, would move
data for each step. That is the idea behind fusing, and the benchmark measures whether it matters.

**Why one program per row?** A softmax row is self-contained, so programs need no synchronization and the row
count sets the amount of parallelism. The block holds exactly one row, which makes the max and sum simple
reductions over a block.

**Why float32 accumulation?** A float16 sum loses small terms next to large ones, because float16 has only about
3 decimal digits. Converting to float32 after the load gives the sum and the division full room, and the final
rounding happens once, at the store.

**Special values.** The result matches `torch.softmax`: a row with any `NaN` or `+inf`, or a row of only `-inf`,
becomes all `NaN`. A `-inf` among finite values becomes 0.

## Online softmax

**The problem.** A row wider than one block cannot be reduced in one step, since it does not fit in a tensor. So
the max and sum have to be built up block by block.

**Derivation.** After block `i`, let `m_i` be the max so far, and let `l_i` be the sum of `exp(x_j - m_i)` over
every column `j` seen so far. Split that sum into old columns and new ones:

```text
l_i = sum_old exp(x_j - m_i)                  + sum_new exp(x_j - m_i)
    = sum_old exp(x_j - m_{i-1}) * exp(m_{i-1} - m_i) + sum_new exp(x_j - m_i)
    = l_{i-1} * exp(m_{i-1} - m_i)            + sum_new exp(x_j - m_i)
```

Together with `m_i = max(m_{i-1}, max(block_i))`, this is the update rule. It starts from `m_0 = -inf` and
`l_0 = 0`. It needs only the two running values, so old blocks do not have to be kept.

**Why it is stable.** `m_{i-1} - m_i` is never positive, and every term of `l_i` is at most 1, because `m_i` is
at least every `x_j` seen. So `l_i` stays between 0 and the number of columns seen.

**The result.** Once all blocks are done, with final values `m` and `l`:

```text
y_j = exp(x_j - m) / l
```

**Memory access.** This needs `m` and `l` from the whole row, so a second pass reads each block again and writes
its output. That makes two reads and one write, against one read and one write for the fused kernel. It is the
cost of rows that do not fit, and the benchmark includes this kernel so that it can be measured.

**The NaN guards.** The textbook update breaks on `-inf`. If a whole block is `-inf`, then `m_i = -inf` and the
exponents are `-inf - (-inf) = NaN`, which poisons `l`, even if the row has finite values later. Two `tl.where`
guards handle this: a `-inf` column adds 0, and the rescale is 1 when the max did not grow. Both test for the
exact case, so a real `NaN` in the input still spreads through the sum, as it does in `torch.softmax`. Without
the guards, `[-inf, -inf, -inf, -inf, 1, 2, -inf]` at `block_size=2` gives all `NaN` instead of
`[0, 0, 0, 0, 0.27, 0.73, 0]`.

## Tiled matmul

For `C = A @ B`, with `A` of shape `(M, K)` and `B` of shape `(K, N)`:

```text
C[tile_m, tile_n] = sum_{tile_k} A[tile_m, tile_k] @ B[tile_k, tile_n]
```

Each program computes one tile of `C`, looping over `K`. A float32 accumulator carries the sum across iterations.
The grid has `ceil(M / BLOCK_M) * ceil(N / BLOCK_N)` programs.

**Memory access.** Each program reads a `BLOCK_M x K` slice of `A` and a `K x BLOCK_N` slice of `B`, then writes one
tile. So all of `A` is read once per column of tiles, and all of `B` is read once per row of tiles. Larger tiles
mean fewer reads overall, but more registers and shared memory per program.

**Why `tl.dot`?** It is the operation the compiler can map to tensor core instructions. What it becomes depends on
the target and the precision. See [COMPILER_DUMPS.md](COMPILER_DUMPS.md#where-tldot-goes):

| Inputs | Result on `sm_80` and newer |
|---|---|
| float32, `ieee` | An unrolled FMA loop: 1024 `fma` per thread for a 64 x 64 x 32 tile |
| float32, `tf32` | `mma` instructions, with inputs rounded to 10 mantissa bits |
| float16 | `mma` instructions |

**Why `ieee` by default?** Triton's default is `tf32`, which is accurate to only about 1e-3, while `torch.matmul`
does not use it by default. The interpreter ignores precision, so the tests there could never reveal the difference.
Choosing the accurate mode makes the default match the comparison, and `tf32` remains an explicit option.

**Limits on `BLOCK_K`.** The CUDA lowering of `tl.dot` needs a K dimension of at least 8 for float32 and 16 for
float16. The interpreter has no such limit, so `tiled_matmul` enforces it in every mode.

**Grouped ordering.** Programs run roughly in id order, so the program-to-tile mapping decides what memory is touched
at around the same time. Row-major order loads a different tile of `B` for each program. With `GROUP_M`, tiles are
walked column by column inside a group of rows:

```text
GROUP_M = 1           GROUP_M = 2
0  1  2  3            0  2  4  6
4  5  6  7            1  3  5  7
8  9  10 11           8  10 12 14
                      9  11 13 15
```

Consecutive programs now read the same tile of `B`, which is more likely to still be in the L2 cache. It changes only
the order, never the result, so it must be a bijection. A test fills the output with `NaN` first, so that a tile left
unwritten cannot pass. Whether it helps depends on the matrix sizes and the L2 size, so it is part of the sweep.

## How block size affects code

From Experiment 2, for the fused softmax with 4 warps (128 threads):

| `BLOCK_SIZE` | Elements per thread | Loads per thread | Registers | Code bytes | Shared bytes |
|---|---|---|---|---|---|
| 16 | under 1 | 1 | 14 | 1152 | 0 |
| 64 | under 1 | 1 | 15 | 1408 | 8 |
| 128 | 1 | 1 | 15 | 1536 | 16 |
| 256 | 2 | 2 | 16 | 1920 | 16 |
| 1024 | 8 | 8 | 38 | 4096 | 16 |

- **Above one element per thread,** each thread needs its own load and registers for each element, so registers and
  code size grow with the block.
- **Below one element per thread,** threads hold duplicates. At 64 on 128 threads, two threads hold each element, and
  the store stops one of them from writing. Those extra threads are not doing useful work.
- **Occupancy** is the fraction of a block that holds real data. A row of 65 columns needs a block of 128, so nearly
  half of it is masked. A block much larger than a row wastes threads and memory on the mask.

For the tiled matmul, the same effect is much larger: at `sm_75`, float16 with a 64 x 64 x 32 tile hits 255 registers,
with 668 bytes of spill stores (Experiment 4). Register pressure is one reason that tile sizes are worth sweeping.
