1) matrix transpose 
2) blocks have x y z dimensions and threads also have x y z dimensions 
3) dim3 threadsPerBlock(16, 16);
   dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
   matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);

4) threads per block 16 * 16 which means it has x and y dimensions ie 16*16  256  threads
5) blocksPerGrid for a 4 rows 6 columns will be just 1 block ie blockIdx.x and blockIdx.y is zero, blockDim is the dimension of threads and hence blockDim.x givs 16 and blockDim.y gives 16
