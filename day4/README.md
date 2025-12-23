<!-- 1) learnt about representation of matrices in GPUS
2) A00 A01 A02 A03
   A10 A11 A12 A13
   A20 A21 A22 A23
3) Lets say we want to do an operation on every single one of them then we would need 12 threads or 4 * 3.
4) We can traverse blocks using x and y like lets say there are 6 blocks 
    B00 B01 B02
    B10 B11 B12
5) In each block lets say we have 10 threads then total we will have 60 threads
6) each block we can give a x and y coordinate lets say we have 2 columns so y can be 0 or 1 and we have 10 threads in each so Y can go from 0 to 20. 
X can go from 0 to 2 and we have like 10 threads each so it can be 0 to 30 -->
