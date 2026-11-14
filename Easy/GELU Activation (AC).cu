#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float *X, float *Y, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
    {
        Y[idx] = (0.5 * X[idx]) * (1.0 + erff(X[idx] / sqrt(2)));
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *X, float *Y, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
