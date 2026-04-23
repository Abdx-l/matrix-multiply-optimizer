#include <algorithm>
#include <vector>
#include <cstddef>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
extern "C" {
void gemm_naive(const float*a,const float*b,float*c,int n){std::fill(c,c+n*n,0);for(int i=0;i<n;i++)for(int j=0;j<n;j++)for(int k=0;k<n;k++)c[i*n+j]+=a[i*n+k]*b[k*n+j];}
void gemm_tiled(const float*a,const float*b,float*c,int n,int tile){
 if(tile<1)tile=32;std::fill(c,c+n*n,0);
 for(int ii=0;ii<n;ii+=tile)for(int kk=0;kk<n;kk+=tile)for(int jj=0;jj<n;jj+=tile)
 for(int i=ii;i<std::min(n,ii+tile);i++)for(int k=kk;k<std::min(n,kk+tile);k++)for(int j=jj;j<std::min(n,jj+tile);j++)c[i*n+j]+=a[i*n+k]*b[k*n+j];
}
void gemm_openmp(const float*a,const float*b,float*c,int n){
 std::fill(c,c+n*n,0);
 #pragma omp parallel for
 for(int i=0;i<n;i++)for(int k=0;k<n;k++)for(int j=0;j<n;j++)c[i*n+j]+=a[i*n+k]*b[k*n+j];
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2,fma")))
void avx_kernel(const float*a,const float*b,float*c,int n){
 std::fill(c,c+n*n,0);
 for(int i=0;i<n;i++)for(int k=0;k<n;k++){auto scalar=_mm256_set1_ps(a[i*n+k]);int j=0;
 for(;j+8<=n;j+=8)_mm256_storeu_ps(c+i*n+j,_mm256_fmadd_ps(scalar,_mm256_loadu_ps(b+k*n+j),_mm256_loadu_ps(c+i*n+j)));
 for(;j<n;j++)c[i*n+j]+=a[i*n+k]*b[k*n+j];}
}
#endif
void gemm_avx2(const float*a,const float*b,float*c,int n){
#if defined(__x86_64__) && defined(__GNUC__)
 if(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")){avx_kernel(a,b,c,n);return;}
#endif
 gemm_tiled(a,b,c,n,32);
}
void gemm_transposed(const float*a,const float*b,float*c,int n){
 std::vector<float> bt(n*n);for(int i=0;i<n;i++)for(int j=0;j<n;j++)bt[j*n+i]=b[i*n+j];
 for(int i=0;i<n;i++)for(int j=0;j<n;j++){float sum=0;for(int k=0;k<n;k++)sum+=a[i*n+k]*bt[j*n+k];c[i*n+j]=sum;}
}
}
