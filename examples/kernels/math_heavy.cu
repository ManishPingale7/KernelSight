// Arithmetic-heavy code whose PTX changes a lot under --use_fast_math:
// division, sqrt and transcendental functions switch to approximate
// hardware instructions (div.approx, sqrt.approx, ex2.approx, ...).

__device__ __noinline__ float smooth(float x)
{
    // __noinline__ keeps this as a separate .func, so the PTX contains a call.
    return x / (1.0f + fabsf(x));
}

extern "C" __global__ void math_heavy(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float y = sqrtf(x * x + 1.0f) / (x + 2.0f);
        y += expf(-x) * sinf(x) + logf(1.0f + x * x);
        out[i] = smooth(y);
    }
}

extern "C" __global__ void polynomial(const double* in, double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in[i];
        double r = 0.0;
        // Horner's rule: a chain of fused multiply-adds (fma.rn.f64).
        const double c[8] = {1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, -0.0078125};
        for (int k = 7; k >= 0; --k) {
            r = r * x + c[k];
        }
        out[i] = r;
    }
}
