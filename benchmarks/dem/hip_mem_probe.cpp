// Standalone reproducer for GitHub issue #807 managed-memory penalties on AMD Instinct.
// Build:
//   hipcc -O3 --offload-arch=gfx950 hip_mem_probe.cpp -o probe && ./probe
//   hipcc -O3 --offload-arch=gfx950:xnack+ hip_mem_probe.cpp -o probe_x && HSA_XNACK=1 ./probe_x

#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void streamKernel(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = a[i] * 1.000001f + b[i];
}
__global__ void atomicKernel(float* acc, const unsigned* owner, const float* val, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(acc + owner[i], val[i]);
}

enum Mode { DEVICE, MANAGED, MANAGED_COARSE };
struct Result {
    float stream_ms;
    float atomic_ms;
    bool advise_ok;
};

static Result bench(Mode mode, size_t n, size_t nOwners, int iters) {
    Result r{};
    r.advise_ok = true;
    float *a = nullptr, *b = nullptr, *acc = nullptr, *val = nullptr;
    unsigned* owner = nullptr;

    auto alloc = [&](void** p, size_t bytes) {
        if (mode == DEVICE) {
            (void)hipMalloc(p, bytes);
        } else {
            (void)hipMallocManaged(p, bytes, hipMemAttachGlobal);
            if (mode == MANAGED_COARSE) {
                hipError_t e = hipMemAdvise(*p, bytes, hipMemAdviseSetCoarseGrain, 0);
                if (e != hipSuccess) {
                    r.advise_ok = false;
                    printf("  hipMemAdvise(SetCoarseGrain) failed: %s\n", hipGetErrorString(e));
                }
            }
        }
    };

    alloc((void**)&a, n * sizeof(float));
    alloc((void**)&b, n * sizeof(float));
    alloc((void**)&val, n * sizeof(float));
    alloc((void**)&owner, n * sizeof(unsigned));
    alloc((void**)&acc, nOwners * sizeof(float));

    std::vector<float> hf(n, 1.0f);
    std::vector<unsigned> ho(n);
    for (size_t i = 0; i < n; i++)
        ho[i] = (unsigned)((i * 7919) % nOwners);
    (void)hipMemcpy(a, hf.data(), n * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(b, hf.data(), n * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(val, hf.data(), n * sizeof(float), hipMemcpyHostToDevice);
    (void)hipMemcpy(owner, ho.data(), n * sizeof(unsigned), hipMemcpyHostToDevice);
    (void)hipMemset(acc, 0, nOwners * sizeof(float));
    (void)hipDeviceSynchronize();

    int threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);

    streamKernel<<<blocks, threads>>>(a, b, n);
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(e0);
    for (int k = 0; k < iters; k++)
        streamKernel<<<blocks, threads>>>(a, b, n);
    (void)hipEventRecord(e1);
    (void)hipDeviceSynchronize();
    (void)hipEventElapsedTime(&r.stream_ms, e0, e1);

    atomicKernel<<<blocks, threads>>>(acc, owner, val, n);
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(e0);
    for (int k = 0; k < iters; k++)
        atomicKernel<<<blocks, threads>>>(acc, owner, val, n);
    (void)hipEventRecord(e1);
    (void)hipDeviceSynchronize();
    (void)hipEventElapsedTime(&r.atomic_ms, e0, e1);

    (void)hipFree(a);
    (void)hipFree(b);
    (void)hipFree(val);
    (void)hipFree(owner);
    (void)hipFree(acc);
    return r;
}

int main() {
    hipDeviceProp_t p;
    (void)hipGetDeviceProperties(&p, 0);
    printf("device: %s  gcnArch: %s\n\n", p.name, p.gcnArchName);

    const size_t n = 8u << 20;
    const size_t nOwners = n / 8;
    const int iters = 50;

    Result d = bench(DEVICE, n, nOwners, iters);
    Result m = bench(MANAGED, n, nOwners, iters);
    Result c = bench(MANAGED_COARSE, n, nOwners, iters);

    printf("%-22s %12s %12s\n", "arm", "stream(ms)", "atomic(ms)");
    printf("%-22s %12.3f %12.3f\n", "hipMalloc device", d.stream_ms / iters, d.atomic_ms / iters);
    printf("%-22s %12.3f %12.3f\n", "managed", m.stream_ms / iters, m.atomic_ms / iters);
    printf("%-22s %12.3f %12.3f%s\n", "managed + coarsegrain", c.stream_ms / iters, c.atomic_ms / iters,
           c.advise_ok ? "" : "  (advise FAILED)");
    printf("\nratios vs device:  managed stream %.1fx atomic %.1fx | coarse stream %.1fx atomic %.1fx\n",
           m.stream_ms / d.stream_ms, m.atomic_ms / d.atomic_ms, c.stream_ms / d.stream_ms, c.atomic_ms / d.atomic_ms);
    return 0;
}
