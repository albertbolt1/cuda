#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    __shared__ int count[256];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    count[tid] = 0;
    if(i<N)
    {
        if(input[i]==K)
        {
            count[tid] = 1;
        }
    }

    __syncthreads();

    for(int l = blockDim.x/2; l>0;l>>=1)
    {
        if(tid<l)
        {
            count[tid]+=count[tid+l];
        }
        __syncthreads();
    }

    if(tid==0)
    {
        atomicAdd(output,count[0]);
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}
