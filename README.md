# Matrix Multiply Optimizer

Naive, tiled, AVX2/FMA and OpenMP float32 GEMM kernels with one ABI, a transposed-B path, tail handling, NumPy correctness checks, and configurable benchmark plots. AVX2 is compiled separately and dispatched at runtime on x86 GCC/Clang.

## Run

Linux: `cmake -S . -B build && cmake --build build`, install requirements.txt, then `python benchmark.py --library build/libgemm.so`. Use `perf stat -e L1-dcache-load-misses python benchmark.py` for hardware counters.
