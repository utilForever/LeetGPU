#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char *image, int width, int height)
{
    for (int i = 0; i < width * height; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            image[4 * i + j] = 255 - image[4 * i + j];
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
