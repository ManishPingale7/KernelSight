// Sum reductions. The shared-memory version builds a tree in shared memory
// with a barrier per step; the warp version uses register shuffles
// (shfl.sync) instead of shared memory.

extern "C" __global__ void reduce_shared(const float* in, float* out, int n)
{
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

extern "C" __global__ void reduce_warp_shuffle(const float* in, float* out, int n)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(out, v);
    }
}
