#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float *input, float *output, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < width * height)
    {
        output[idx] = 0.299 * input[3 * idx] + 0.587 * input[3 * idx + 1] + 0.114 * input[3 * idx + 2];
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int width, int height)
{
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
