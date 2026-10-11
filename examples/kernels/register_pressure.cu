// Register allocation happens in ptxas, after PTX. PTX uses an unlimited
// set of virtual registers; ptxas maps them onto the physical registers of
// the target GPU. When a register limit is too low (ptxas -maxrregcount),
// ptxas "spills" values to local memory and reports spill stores/loads in
// its -v output. Those spills are not visible in the PTX itself.

#define N 32

extern "C" __global__ void many_live_values(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        acc[k] = in[(i + k * 97) % n];
    }
    float total = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int k = 0; k < N; ++k) {
            total += acc[k] * acc[(k + r + 1) % N];
        }
    }
    out[i] = total;
}

// Indexing a local array with a runtime value forces it into local memory,
// so ld.local / st.local appear directly in the PTX.
extern "C" __global__ void dynamic_local_index(const int* idx, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float table[16];
    for (int k = 0; k < 16; ++k) {
        table[k] = k * 0.5f + i;
    }
    out[i] = table[idx[i] & 15];
}
