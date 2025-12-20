1) learnt about the very basics that there are blocks and that blocks have threads 
2) we want to do massively parallel computation so we will do the computation like this 
thread 0 - C[0] = A[0] + B[0];
thread 1 - C[1] = A[1] + B[1];
etc 
3) Kernel is basically just a program executing in GPU 
4) __global__ will execute in GPU called by CPU
5) __device__ will be executed in GPU called by GPU
6) __host__ will be executed in CPU called by CPU

Also revised some general c++ stuff 
1) float * A is a pointer to an array 
2) A holds the address of the array
3) we can get the next element by doing *(A+1);

