// Histogram with a per-block copy in shared memory ("privatization") and
// atomics, so most atomic traffic stays in fast shared memory.

#define BINS 256

extern "C" __global__ void histogram256(const unsigned char* data, unsigned* hist, int n)
{
    __shared__ unsigned local_hist[BINS];
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        local_hist[b] = 0;
    }
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        atomicAdd(&local_hist[data[i]], 1u);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < BINS; b += blockDim.x) {
        atomicAdd(&hist[b], local_hist[b]);
    }
}
