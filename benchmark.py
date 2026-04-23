import argparse,ctypes,time,csv
import numpy as np
def run(library,sizes=(7,16,31,64,128)):
 lib=ctypes.CDLL(library);rng=np.random.default_rng(42);results=[]
 ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
 for n in sizes:
  a=rng.random((n,n),dtype=np.float32);b=rng.random((n,n),dtype=np.float32);expected=a@b;c=np.empty_like(a)
  for name in ("naive","tiled","avx2","openmp","transposed"):
   function=getattr(lib,"gemm_"+name);function.argtypes=[ptr,ptr,ptr,ctypes.c_int]+([ctypes.c_int] if name=="tiled" else [])
   for tile in ((16,32,64,128) if name=="tiled" else (0,)):
    args=(a,b,c,n,tile) if name=="tiled" else (a,b,c,n);function(*args)
    np.testing.assert_allclose(c,expected,rtol=2e-5,atol=1e-5)
    start=time.perf_counter();function(*args);elapsed=time.perf_counter()-start
    results.append(dict(n=n,kernel=name,tile=tile,gflops=2*n**3/elapsed/1e9))
 return results
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--library",default="build/libgemm.so");a=p.parse_args();rows=run(a.library)
 with open("benchmark.csv","w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 import matplotlib.pyplot as plt
 for kernel in {r["kernel"] for r in rows}:
  points=[r for r in rows if r["kernel"]==kernel and r["tile"] in (0,32)];plt.plot([r["n"] for r in points],[r["gflops"] for r in points],label=kernel)
 plt.xlabel("Matrix dimension");plt.ylabel("GFLOPS");plt.legend();plt.savefig("benchmark.png");print("All kernels match NumPy")
