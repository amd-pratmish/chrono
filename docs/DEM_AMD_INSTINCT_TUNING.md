# Chrono::DEM on AMD Instinct (MI300X / MI350X)

This note documents the performance fix for [GitHub issue #807](https://github.com/projectchrono/chrono/issues/807): `chrono_dem` was up to **28× slower per step** on MI350X than on RTX 5090 for dense granular contact workloads.

## Root cause

1. **Managed memory on `xnack-` builds** — particle arrays used `hipMallocManaged`. With default `--offload-arch=gfx950` (xnack disabled), pages do not migrate into HBM; `atomicAdd` on accelerations sees **system-scope** coherence and can be **>1000× slower** than `hipMalloc` (see standalone probe below).

2. **Host touches during the time step** — even with xnack+, host reads/writes to managed arrays (BC updates, I/O, queries) can pull pages back to host memory each step, reintroducing migration cost.

3. **Not a pure compile-flag issue** — rebuilding only with `xnack+` recovered ~2–3×, not the full application gap; allocation policy and host access patterns matter.

## Fix (this branch)

| Component | CUDA (unchanged) | HIP (MI Instinct) |
|-----------|------------------|-------------------|
| Bulk particle arrays (`gpuallocator`) | `cudaMallocManaged` | **`hipMalloc` (device memory)** |
| Mesh node arrays | managed | **`hipMalloc` via `demGpuMallocBulk`** |
| Small structs (`gran_params`, `sphere_data`) | managed | managed + **coarse-grain advise + prefetch** |
| BC force reset | host loop | **device kernel** |
| BC position update | host writes to managed vector | **host shadow + H2D copy** |
| AdvanceSimulation | reads `gran_params` each step | **cached friction/integrator flags** |

Override (debug only): set `CHRONO_DEM_HIP_MANAGED=1` to restore managed bulk allocations on HIP.

## Correctness fixes (157k-grain crash)

Full-scale runs (~157k grains) failed in `integrateSpheres` with `negative local pos in SD` and `HSA_STATUS_ERROR_EXCEPTION`.

**Root causes:**

1. **Subdomain index used truncating division** — when a sphere's BD-relative coordinate went slightly negative (bottom layer / SD boundary crossing), C++ `/` mapped it to SD index 0 with a negative local offset, triggering a device abort in `findNewLocalCoords`. Fixed with floor division (`demDivFloorSD`) in `pointSDTriplet` and matching host paths.

2. **Managed `gran_params` / `sphere_data` not visible on GPU under xnack-** — `hipMemPrefetchAsync` alone could leave stale struct fields on device during the first integration steps. `demGpuPublishManagedToDevice` now uses explicit `hipMemcpy` H2D when the HIP device-memory bulk policy is active.

3. **BC device sync churn** — `syncBCParamsListToDevice` reallocated every call; now reuses device buffers and only grows when needed.

After these fixes, re-run `demo_DEM_settling_bench --grains 159600 --steps 25000` and compare **ns/grain/step** vs RTX 5090 (7.1 ns target).

## Build

```bash
cmake -DCH_ENABLE_MODULE_DEM=ON -DCHRONO_GPU_BACKEND=HIP \
      -DCHRONO_HIP_ARCHITECTURES=gfx950 \  # or gfx942 for MI300X
      -DCMAKE_BUILD_TYPE=Release ..
ninja Chrono_dem demo_DEM_settling_bench
```

Fat binary (optional): pass both `--offload-arch=gfx950:xnack+` and `--offload-arch=gfx950:xnack-`; full speed under `HSA_XNACK=1`, still runs under `HSA_XNACK=0`.

## Standalone memory probe

From `benchmarks/dem/`:

```bash
hipcc -O3 --offload-arch=gfx950 hip_mem_probe.cpp -o probe && ./probe
```

Expect ~1× device vs managed+coarse on xnack+ with `HSA_XNACK=1`, and large managed penalties on xnack-.

## Application benchmark

```bash
./bin/demo_DEM_settling_bench --grains 159600 --steps 25000
```

Reports **ns/grain/step** (lower is better). Compare against a CUDA 5090 build of the same demo for regression tracking.

## Profiling

For remaining gaps after this patch:

```bash
rocprofv3 --kernel-trace --stats -- ./bin/demo_DEM_settling_bench --grains 18870 --steps 100
```

Inspect `computeSphereContactForces_matBased` achieved bandwidth and atomic stall counters.
