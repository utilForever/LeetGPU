#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char *image, int width, int height)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < width * height)
    {
        for (int i = 0; i < 3; ++i)
        {
            image[4 * idx + i] = 255 - image[4 * idx + i];
        }
    }
}

// image is a device pointer (i.e. pointer to memory on the GPU)
extern "C" void solve(unsigned char *image, int width, int height)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
