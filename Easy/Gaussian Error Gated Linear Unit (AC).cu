#include <cuda_runtime.h>

__global__ void geglu_kernel(const float *input, float *output, int halfN)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < halfN)
    {
        float gelu = 0.5 * input[idx + halfN] * (1.0 + erff(input[idx + halfN] / sqrt(2)));
        output[idx] = input[idx] * gelu;
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N)
{
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
