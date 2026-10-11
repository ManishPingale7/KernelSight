// Simple element-wise kernels: one load/compute/store per element.
// These are memory-bound: each element is read and written once.

extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void saxpy(float alpha, const float* __restrict__ x,
                                 float* __restrict__ y, int n)
{
    // Grid-stride loop: each thread handles several elements.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = alpha * x[i] + y[i];
    }
}

// float4 loads/stores compile to ld.global.v4.f32 / st.global.v4.f32.
extern "C" __global__ void scale_vec4(const float4* __restrict__ in, float4* __restrict__ out,
                                      float s, int n4)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        out[i] = make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
    }
}
