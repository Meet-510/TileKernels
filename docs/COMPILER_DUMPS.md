# Reading the compiler dumps

This follows one kernel, the fused softmax, through each stage of the Triton compiler. Every excerpt
below comes from a file in [`dumps/`](../dumps), and `tests/test_docs.py` checks that each one is still
there. The counts come from [`dumps/experiments/results.md`](../dumps/experiments/results.md).

## How these dumps were made

No GPU was involved. `triton.compile` takes an explicit `GPUTarget`, and the wheel bundles `ptxas`, so
compilation works on any machine.

```bash
python scripts/dump_ir.py --all
```

```bash
python scripts/ir_experiments.py
```

**Provenance:** the committed dumps were compiled on Windows with `triton-windows` 3.9.0.post30 for
`sm_86`, with line info kept and local paths replaced by `<repo>` and `<triton>`. The DWARF tables at the
end of each PTX file are removed, because they hold the checkout path as raw bytes. Generated code may
differ on other versions. The kernels here have not yet run on real GPU hardware, so this document
describes what the compiler generates, and not how fast it runs.

### What could and could not be produced without a GPU

| Stage | Without a GPU? | How |
|---|---|---|
| TTIR, TTGIR, LLIR, PTX | Yes | `compiled.asm` dict, written by `scripts/dump_ir.py` |
| Cubin | Yes | `compiled.asm["cubin"]`, generated but not committed |
| Registers and spills | Yes | `ptxas -v` log, with `TRITON_DUMP_PTXAS_LOG=1` |
| Shared memory | Yes | `compiled.metadata.shared` |
| IR after every pass | Yes | `MLIR_ENABLE_DUMP=1` |
| SASS disassembly | **Not here** | Needs `cuobjdump` and `nvdisasm`, which the Windows wheel lacks |

`TRITON_KERNEL_DUMP=1` also writes every stage, to `TRITON_DUMP_DIR`. On Windows without the CUDA tools,
it writes the stages and then crashes while disassembling, so the script reads `.asm` instead.

To get the IR after every compiler pass, redirect stderr. This is verbose: 7 MB for the matmul kernel.

```bash
MLIR_ENABLE_DUMP=1 python scripts/dump_ir.py --kernel fused_softmax 2> passes.log
```

## The kernel

`BLOCK_SIZE=64`, `num_warps=4`, `sm_86`. The source is
[`_fused_softmax_kernel`](../src/tilekernels/softmax.py).

## Stage 1: TTIR

TTIR is the first MLIR form, made directly from the Python AST. It uses tensors of block size, and says
nothing about threads. Constexprs are already gone: `BLOCK_SIZE` has become the `64` in each type.

The mask is an ordinary `tensor<64xi1>` value, a comparison between the column range and `n_cols`. It is
then the second operand of `tt.load`, and the third operand is the `other` value:

<!-- excerpt: dumps/fused_softmax/fused_softmax.ttir -->
```text
%x = arith.constant dense<0xFF800000> : tensor<64xf32>
%in_row = tt.splat %n_cols : i32 -> tensor<64xi32>
%in_row_0 = arith.cmpi slt, %cols, %in_row : tensor<64xi32>
%x_7 = tt.load %x_6, %in_row_0, %x : tensor<64x!tt.ptr<f32>>
```

`0xFF800000` is float32 `-inf`, from `other=-float("inf")`. The same mask value is reused by the store.

`tl.max` and `tl.sum` have been inlined from Triton's standard library into `tt.reduce` regions:

<!-- excerpt: dumps/fused_softmax/fused_softmax.ttir -->
```text
%numerator = "tt.reduce"(%x_7) <{axis = 0 : i32}> ({
%numerator_20 = arith.maxnumf %numerator_18, %numerator_19 : f32
```

## Stage 2: TTGIR

TTGIR is the same program, with one big addition: each tensor now has a layout, which says which thread
holds which element. Everything else lines up with TTIR almost line for line.

<!-- excerpt: dumps/fused_softmax/fused_softmax.ttgir -->
```text
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
%x_6 = tt.load %x_5, %in_row_0, %cst : tensor<64x!tt.ptr<f32>, #blocked>
```

Layouts are assigned by the `ConvertTritonToTritonGPU` pass, which is where `num_warps` first takes
effect. In a pass trace for the matmul kernel, the layout count goes from 0 to 19 at that point, and the
tensor core layout only appears after `TritonGPUAccelerateMatmul` (see below).

## Stage 3: LLVM IR

Tensors are gone. Each thread now runs scalar code, and the block is implicit in the thread id. Here
`and i32 %11, 63` turns the thread id into a column index, and the mask has become an `i1`:

<!-- excerpt: dumps/fused_softmax/fused_softmax.llir -->
```text
%11 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%12 = and i32 %11, 63
%13 = icmp slt i32 %12, %2
```

### The reduction becomes warp shuffles

`tl.max` over 64 elements has no single instruction, so it becomes a butterfly: each thread swaps its
value with a partner at distance 16, 8, 4, 2, then 1, and keeps the larger of the two.

<!-- excerpt: dumps/fused_softmax/fused_softmax.llir -->
```text
%22 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %20, i32 16, i32 31)
%24 = tail call float @llvm.maxnum.f32(float %21, float %23)
%26 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %25, i32 8, i32 31)
%30 = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %29, i32 4, i32 31)
```

A shuffle only reaches threads in the same warp. To combine across warps, the code goes through shared
memory, with a barrier in between. That is why this kernel needs 8 bytes of shared memory:

<!-- excerpt: dumps/fused_softmax/fused_softmax.llir -->
```text
store float %40, ptr addrspace(3) %43, align 4
tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
%47 = load i32, ptr addrspace(3) %46, align 4
```

### exp is an approximation

`tl.exp` becomes a base 2 exponential, scaled by log2(e), which is 0x3FB8AA3B as a float:

<!-- excerpt: dumps/fused_softmax/fused_softmax.llir -->
```text
%53 = fmul float %52, f0x3FB8AA3B
%54 = tail call float @llvm.nvvm.ex2.approx.f32(float %53)
```

The interpreter uses NumPy's `exp` instead, so its accuracy tells you nothing about this instruction.
That is a reason to also run `pytest --gpu` on real hardware.

## Stage 4: PTX

PTX is close to LLVM IR, but with named registers. `.loc` directives point back at Python source lines,
and `softmax.py:46` is the `tl.load`.

### The mask on a load

This is the clearest picture of what a masked load is. First the `other` value is moved into the
destination register, where `-8388608` is `0xFF800000` as a signed integer. Then the load overwrites it,
but only in threads where predicate `%p1`, the mask, is true:

<!-- excerpt: dumps/fused_softmax/fused_softmax.ptx -->
```text
setp.lt.s32 	%p1, %r7, %r5;
mov.b32 	%r2, -8388608;
mov.u32 %r1, %r2;
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ];
```

A masked thread does no memory access at all, which is why reading past the end of a row is safe.

### tl.load does not become a vector load here

You might expect `tl.load` to become a vector instruction. In these kernels, it does not. There is no
vector anywhere in this PTX: the "vector" is the group of threads, each running the same scalar
`ld.global.b32` on its own element. That is the SIMT model. See the
[first debugging story](#story-1-my-guess-about-vector-loads-was-wrong) for what happens when each thread
holds more than one element.

### The reduction, in PTX

<!-- excerpt: dumps/fused_softmax/fused_softmax.ptx -->
```text
shfl.sync.bfly.b32 	%r14, %r1, 16, 31, -1;
max.f32 	%r15, %r1, %r14;
shfl.sync.bfly.b32 	%r16, %r15, 8, 31, -1;
st.shared.b32 	[%r27], %r23;
bar.sync 	0;
```

## Stage 5: Cubin metadata

`dumps/fused_softmax/metadata.json` holds what the assembler and compiler report: 15 registers, no spills,
1408 bytes of machine code and 8 bytes of shared memory.

## Where tl.dot goes

The softmax has no `tl.dot`, so these excerpts are from the matmul dumps. In TTIR, a `tt.dot` is one
operation. What it becomes depends on precision and target.

**IEEE float32: an FMA loop.** Tensor cores do not support full float32 inputs, so the dot is expanded
into one fused multiply-add per element per K step. This is why `dumps/tiled_matmul_fma` uses a tile of
16 x 16 x 8: at 64 x 64 x 32, there are 1024 FMA instructions per thread.

<!-- excerpt: dumps/tiled_matmul_fma/tiled_matmul_fma.ttgir -->
```text
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
```

<!-- excerpt: dumps/tiled_matmul_fma/tiled_matmul_fma.ptx -->
```text
fma.rn.f32 	%r97, %r72, %r89, %r129;
fma.rn.f32 	%r98, %r73, %r90, %r97;
```

**TF32 float32: an MMA instruction.** `TritonGPUAccelerateMatmul` replaces the layout with a tensor core
layout, whose `instrShape` says that one instruction computes a 16 x 8 piece of the result:

<!-- excerpt: dumps/tiled_matmul/tiled_matmul.ttgir -->
```text
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
```

<!-- excerpt: dumps/tiled_matmul/tiled_matmul.ptx -->
```text
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
```

Since one instruction consumes 8 values of K, a `BLOCK_K` of 32 becomes four `tt.dot` operations per
iteration in TTGIR.

**Loads become asynchronous copies.** With tensor cores, the operands live in shared memory.
`tl.load` becomes an async copy of 4 bytes per element, and software pipelining starts the next tile's
copies while the current tile is being multiplied. `num_stages` sets how many tiles are in flight.

<!-- excerpt: dumps/tiled_matmul/tiled_matmul.ptx -->
```text
cp.async.ca.shared.global [ %r94 + 0 ], [ %rd19 + 0 ], 0x4, %r95;
cp.async.commit_group;
cp.async.wait_group 	2;
```

Across targets, the same `tl.dot` becomes:

| Target | float32 `ieee` | float32 `tf32` | float16 |
|---|---|---|---|
| `sm_75` | FMA | FMA | FMA |
| `sm_80`, `sm_86` | FMA | `mma.sync` | `mma.sync` |
| `sm_90` | FMA | `wgmma` | `wgmma` |
| `sm_100` | FMA | `tcgen05.mma` | `tcgen05.mma` |

`tests_compile/test_compile_for_gpu.py` pins this table.

## Debugging stories

All of these came from real runs, and can be reproduced with the commands above.

### Story 1: my guess about vector loads was wrong

**The question.** `fused_softmax` takes the column stride as a runtime argument, so that it can support
non-contiguous inputs. I expected this to have a cost: with a stride unknown at compile time, surely the
compiler could not merge neighbouring elements into a vector load.

**The test.** In `scripts/ir_experiments.py`, I compiled the kernel with the stride fixed at 1, and again
with no mask, at `BLOCK_SIZE=1024` so that each of the 128 threads holds 8 elements. From
[Experiment 3](../dumps/experiments/results.md):

| Variant | Registers | Code bytes | Global loads | Predicated |
|---|---|---|---|---|
| Runtime stride, masked | 38 | 4096 | 8 x `ld.global.b32` | 8 |
| Stride fixed at 1, masked | 29 | 3200 | 8 x `ld.global.b32` | 8 |
| Stride fixed at 1, no mask | 25 | 2816 | 8 x `ld.global.b32` | 0 |

**The surprise.** There are still 8 scalar loads in every variant, so my guess was wrong.

**The explanation, from the dump.** The layout has `sizePerThread = [1]`, so elements are dealt out to
threads one at a time, like cards. In the PTX for block 1024, each thread computes its 8 column indices
by setting high bits of its thread id: `%tid & 127`, `| 128`, `| 256`, up to `| 896`. A thread's elements
are 128 apart in memory, so there is nothing contiguous to merge, whatever the stride is.

**What it does cost.** The runtime stride makes the code 28% larger here (4096 bytes against 3200), since
each element's address needs a multiply. The mask makes it 14% larger (3200 against 2816), for the eight
`setp` comparisons.
[Experiment 6](../dumps/experiments/results.md) shows that in the matmul kernel, runtime strides change
nothing at all on `sm_86`: registers, code size and the async copy pattern are all identical.

**The limit of this.** These are code size counts, not timings. Whether the extra instructions matter
needs a GPU.

### Story 2: the interpreter accepted a kernel the compiler rejects

The interpreter tests passed with `BLOCK_K=1`. Compiling the same kernel failed:

```text
Input shapes should have M >= 1, N >= 1 and K >= 8
```

The interpreter implements `tl.dot` with `np.matmul`, so it has no such limit, while the CUDA lowering
does: at least 8 for float32 and 16 for float16, on `sm_80` and `sm_90` where I mapped the boundary. A
config could therefore pass the whole test suite and still fail on a GPU. So `tiled_matmul` now rejects it up front in every mode,
and `tests_compile/` checks that the limit is still what the compiler enforces.

### Story 3: num_stages does nothing on a T4

The backend only runs its software pipeliner on `sm_80` and newer, so I checked what `num_stages` does on
`sm_75`. From [Experiment 5](../dumps/experiments/results.md), for a 64 x 64 x 32 tile:

| Target | `num_stages` | Shared bytes | Code bytes | Same PTX as `num_stages=1` |
|---|---|---|---|---|
| `sm_75` | 1 to 4 | 16384 | 28416 | yes |
| `sm_86` | 1 | 16384 | 12928 | yes |
| `sm_86` | 2 | 16384 | 18816 | no |
| `sm_86` | 3 | 32768 | 23680 | no |
| `sm_86` | 4 | 49152 | 27776 | no |

So on the free T4, sweeping `num_stages` measures only noise, and all differences in the results would be
measurement error. On `sm_86`, each stage from 3 onward buffers one more set of tiles: one tile each of A
and B is 64 x 32 x 4 + 32 x 64 x 4 = 16384 bytes.

### Story 4: more warps can add threads with no work

For `BLOCK_SIZE=64`, the code is the same size at 2, 4 and 8 warps
([Experiment 1](../dumps/experiments/results.md)). With 4 warps there are 128 threads for 64 elements,
so the column index is `tid & 63`, and two threads hold each element. The LLVM IR shows how the store
avoids writing twice, by only letting threads with bit 6 clear write:

<!-- excerpt: dumps/fused_softmax/fused_softmax.llir -->
```text
%87 = and i32 %11, 64
%88 = icmp eq i32 %87, 0
%90 = and i1 %88, %13
```

So half the threads in this configuration do redundant work, and `sm_86` needs a barrier and shared memory
to combine reductions across warps that one warp would not need. With `num_warps=1`, the kernel has no
barrier and no shared memory, though each thread then holds 2 elements and does 2 loads. Which of these
is faster is a question for the benchmark.

## Optional: disassembly on a machine with CUDA

`cuobjdump` and `nvdisasm` come with the CUDA toolkit. If Triton can find them, for example through
`TRITON_CUOBJDUMP_PATH` and `TRITON_NVDISASM_PATH`, then `scripts/dump_ir.py` also writes a `.sass` file.
This path has not been tested, because the tools were not available where this project was built.
