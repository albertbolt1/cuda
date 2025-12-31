#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int i= blockDim.x * blockIdx.x + threadIdx.x;

        if(j<M && i<K)
        {
            float temp=0.0f;
            for(int x=0;x<N;x++)
            {
                temp+=A[j*N+x]*B[x*K+i];
            }
            C[j*K+i]=temp;
        }
        }

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
