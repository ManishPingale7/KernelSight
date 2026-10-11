// Matrix transpose: naive (uncoalesced writes) vs a shared-memory tile with
// one column of padding to avoid shared-memory bank conflicts.

#define TDIM 32

extern "C" __global__ void transpose_naive(const float* in, float* out, int width, int height)
{
    int x = blockIdx.x * TDIM + threadIdx.x;
    int y = blockIdx.y * TDIM + threadIdx.y;
    if (x < width && y < height) {
        out[x * height + y] = in[y * width + x];
    }
}

extern "C" __global__ void transpose_tiled(const float* in, float* out, int width, int height)
{
    __shared__ float tile[TDIM][TDIM + 1];  // +1 column of padding
    int x = blockIdx.x * TDIM + threadIdx.x;
    int y = blockIdx.y * TDIM + threadIdx.y;
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();
    x = blockIdx.y * TDIM + threadIdx.x;
    y = blockIdx.x * TDIM + threadIdx.y;
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
