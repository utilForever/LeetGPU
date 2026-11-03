#include <cuda_runtime.h>

__global__ void vector_add(const float *A, const float *B, float *C, int N)
{
    // blockDim: Tells how many threads are in a block (in each dimension)
    // blockIdx: Identifies the block within the grid
    // threadIdx: Identifies the thread within a block
    // Reference: https://www.kapilsharma.dev/posts/ai-gpu-programming-1/
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
