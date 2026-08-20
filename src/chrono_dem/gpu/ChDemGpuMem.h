// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2026 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// AMD Instinct / HIP memory policy for Chrono::DEM (GitHub issue #807).
//
// On MI300X/MI350X with xnack-, hipMallocManaged incurs catastrophic penalties
// for atomicAdd-heavy contact kernels.  Bulk particle arrays therefore use
// hipMalloc (device memory) on HIP while CUDA keeps the existing managed path.
// Small host-visible structs may remain managed with coarse-grain advice.
// =============================================================================

#pragma once

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "chrono_dem/ChDemDefines.h"

namespace chrono {
namespace dem {

/// True when gpuallocator and bulk arrays should use device memory (HIP path).
inline bool demGpuUsesDeviceMemory() {
#if defined(CHRONO_USE_HIP)
    static const bool force_managed = []() {
        const char* env = std::getenv("CHRONO_DEM_HIP_MANAGED");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return !force_managed;
#else
    return false;
#endif
}

inline int demGpuDeviceId() {
    int dev = 0;
    (void)gpuGetDevice(&dev);
    return dev;
}

/// Push host-written managed regions to the current GPU (call after CPU fills gran_params/sphere_data).
inline void demGpuPublishManagedToDevice(const void* ptr, std::size_t bytes) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
#if defined(CHRONO_USE_HIP)
    if (demGpuUsesDeviceMemory()) {
        // On xnack- Instinct builds, prefetch alone may not expose host writes to device kernels in time.
        demErrchk(gpuMemcpy(const_cast<void*>(ptr), ptr, bytes, gpuMemcpyHostToDevice));
        return;
    }
#endif
    const int dev = demGpuDeviceId();
    (void)gpuMemPrefetchAsync(ptr, bytes, dev, 0);
}

/// Apply coarse-grain advice and prefetch managed regions to the current GPU.
inline void demGpuAdviseManagedRegion(const void* ptr, std::size_t bytes) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }
    const int dev = demGpuDeviceId();
    (void)gpuMemAdvise(ptr, bytes, gpuMemAdviseSetCoarseGrain, dev);
    (void)gpuMemPrefetchAsync(ptr, bytes, dev, 0);
}

inline gpuError demGpuMallocManaged(void** ptr, std::size_t bytes, unsigned int flags = gpuMemAttachGlobal) {
    gpuError err = gpuMallocManaged(ptr, bytes, flags);
    if (err == gpuSuccess && *ptr != nullptr) {
        demGpuAdviseManagedRegion(*ptr, bytes);
    }
    return err;
}

template <class T>
inline gpuError demGpuMallocManaged(T** ptr, std::size_t bytes, unsigned int flags = gpuMemAttachGlobal) {
    return demGpuMallocManaged(reinterpret_cast<void**>(ptr), bytes, flags);
}

/// Bulk GPU arrays touched almost exclusively by kernels (mesh nodes, particle state, etc.).
inline gpuError demGpuMallocBulk(void** ptr, std::size_t bytes) {
    if (demGpuUsesDeviceMemory()) {
        return gpuMalloc(ptr, bytes);
    }
    return demGpuMallocManaged(ptr, bytes);
}

template <class T>
inline gpuError demGpuMallocBulk(T** ptr, std::size_t bytes) {
    return demGpuMallocBulk(reinterpret_cast<void**>(ptr), bytes);
}

/// Read one element from a device-resident array when bulk storage is not host-accessible.
template <class T>
inline T demGpuReadElement(const T* device_ptr, std::size_t index) {
#if defined(CHRONO_USE_HIP)
    if (demGpuUsesDeviceMemory()) {
        T value{};
        demErrchk(gpuMemcpy(&value, device_ptr + index, sizeof(T), gpuMemcpyDeviceToHost));
        return value;
    }
#endif
    return device_ptr[index];
}

template <class T>
inline void demGpuWriteElement(T* device_ptr, std::size_t index, const T& value) {
#if defined(CHRONO_USE_HIP)
    if (demGpuUsesDeviceMemory()) {
        demErrchk(gpuMemcpy(device_ptr + index, &value, sizeof(T), gpuMemcpyHostToDevice));
        return;
    }
#endif
    device_ptr[index] = value;
}

template <class T>
inline T demGpuReadScalar(const T* device_ptr) {
#if defined(CHRONO_USE_HIP)
    if (demGpuUsesDeviceMemory()) {
        T value{};
        demErrchk(gpuMemcpy(&value, device_ptr, sizeof(T), gpuMemcpyDeviceToHost));
        return value;
    }
#endif
    return *device_ptr;
}

template <class T>
inline void demGpuCopyToHost(const T* device_ptr, T* host_ptr, std::size_t count) {
    if (count == 0) {
        return;
    }
#if defined(CHRONO_USE_HIP)
    if (demGpuUsesDeviceMemory()) {
        demErrchk(gpuMemcpy(host_ptr, device_ptr, count * sizeof(T), gpuMemcpyDeviceToHost));
        return;
    }
#endif
    std::memcpy(host_ptr, device_ptr, count * sizeof(T));
}

template <class T>
inline void demGpuCopyFromHost(T* device_ptr, const T* host_ptr, std::size_t count) {
    if (count == 0) {
        return;
    }
#if defined(CHRONO_USE_HIP)
    if (demGpuUsesDeviceMemory()) {
        demErrchk(gpuMemcpy(device_ptr, host_ptr, count * sizeof(T), gpuMemcpyHostToDevice));
        return;
    }
#endif
    std::memcpy(device_ptr, host_ptr, count * sizeof(T));
}

}  // namespace dem
}  // namespace chrono
