#include <cuda_runtime.h>

__global__ void matrix_add(const float *A, const float *B, float *C, int N)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < N && col < N)
    {
        int idxCurr = row * N + col;
        int idxNext = (row + 1) * N + col;

        for (int idx = idxCurr; idx < idxNext; ++idx)
        {
            C[idx] = A[idx] + B[idx];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
