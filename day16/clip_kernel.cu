#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<N)
    {
        float temp=input[i];
        if(temp<lo)
        {
            output[i]=lo;
        }
        else if(temp>hi)
        {
            output[i]=hi;
        }
        else
        {
            output[i]=temp;
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
