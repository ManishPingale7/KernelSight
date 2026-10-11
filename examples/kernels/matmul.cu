// Two matrix multiplications: a naive version and a shared-memory tiled one.
// Comparing their PTX shows shared-memory traffic (ld.shared) and barriers
// (bar.sync) replacing repeated global loads.

#define TILE 16

extern "C" __global__ void matmul_naive(const float* A, const float* B, float* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int n)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}
