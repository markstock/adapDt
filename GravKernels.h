/*
 * GravKernels.h - nbody kernels for gravitational forces
 *
 * Part of adapDt - testing hierarchical time stepping for gravitation simulations
 *
 * Copyright (c) 2023-4 Mark J Stock <markjstock@gmail.com>
*/
#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cassert>

#ifdef USE_MPI
#include <mpi.h>
#endif

static float num_flops_kernel = 22.f;
static float num_flops_accjerk = 23.f;
static float num_flops_jerk = 17.f;

//
// kernels to compute accelerations on particles
//

// serial (x86) instructions
template <class T>
static inline void nbody_kernel(const T sx, const T sy, const T sz,
                                const T sm, const T sr,
                                const T tx, const T ty, const T tz, const T tr2,
                                T* const __restrict__ tax, T* const __restrict__ tay, T* const __restrict__ taz) {
    // 22 flops
    const T dx = sx - tx;
    const T dy = sy - ty;
    const T dz = sz - tz;
    const T r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr2;
    const T invR = 1.0 / sqrt(r2);
    const T invR2 = invR * invR;
    const T factor = sm * invR * invR2;
    *tax += factor * dx;
    *tay += factor * dy;
    *taz += factor * dz;
}
// specialize for float
template <>
inline void nbody_kernel(const float sx, const float sy, const float sz,
                         const float sm, const float sr,
                         const float tx, const float ty, const float tz, const float tr2,
                         float* const __restrict__ tax, float* const __restrict__ tay, float* const __restrict__ taz) {
    // 22 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    const float dz = sz - tz;
    const float r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr2;
    const float invR = 1.0f / sqrtf(r2);
    const float invR2 = invR * invR;
    const float factor = sm * invR * invR2;
    *tax += factor * dx;
    *tay += factor * dy;
    *taz += factor * dz;
}

template <class T>
void __attribute__ ((noinline)) nbody_serial(const int ns_start, const int ns_end,
                                     const T* const __restrict__ x,
                                     const T* const __restrict__ y,
                                     const T* const __restrict__ z,
                                     const T* const __restrict__ m,
                                     const T* const __restrict__ r,
                                     const int nt_start, const int nt_end,
                                     T* const __restrict__ ax,
                                     T* const __restrict__ ay,
                                     T* const __restrict__ az) {

    #pragma omp parallel for
    for (int i = nt_start; i < nt_end; i++) {
        const T tr2 = r[i]*r[i];
        for (int j = ns_start; j < ns_end; j++) {
            nbody_kernel<T>(x[j], y[j], z[j], m[j], r[j],
                            x[i], y[i], z[i], tr2, &ax[i], &ay[i], &az[i]);
        }
    }
}

#include "VectorSoA.h"

template <class T, class IT>
void nbody_serial(const int ns_start, const int ns_end,
                  const VectorSoA<T,IT,3>& pos,
                  const std::vector<T>& m,
                  const std::vector<T>& r,
                  const int nt_start, const int nt_end,
                  VectorSoA<T,IT,3>& acc) {

    nbody_serial(ns_start, ns_end,
                 pos.x[0].data(), pos.x[1].data(), pos.x[2].data(),
                 m.data(), r.data(),
                 nt_start, nt_end,
                 acc.x[0].data(), acc.x[1].data(), acc.x[2].data());
}


//
// kernels to compute accelerations and jerk magnitude on particles
//

// serial (x86) instructions
template <class T>
static inline void nbody_kernel_accjerk(const T sx, const T sy, const T sz,
                                const T sm, const T sr,
                                const T tx, const T ty, const T tz, const T tr2,
                                T* const __restrict__ tax, T* const __restrict__ tay, T* const __restrict__ taz,
                                T* const __restrict__ tjm) {
    // 22 flops
    const T dx = sx - tx;
    const T dy = sy - ty;
    const T dz = sz - tz;
    const T r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr2;
    const T invR = 1.0 / sqrt(r2);
    const T invR2 = invR * invR;
    const T factor = sm * invR * invR2;
    *tax += factor * dx;
    *tay += factor * dy;
    *taz += factor * dz;
    *tjm += factor;
}
// specialize for float
template <>
inline void nbody_kernel_accjerk(const float sx, const float sy, const float sz,
                         const float sm, const float sr,
                         const float tx, const float ty, const float tz, const float tr2,
                         float* const __restrict__ tax, float* const __restrict__ tay, float* const __restrict__ taz,
                         float* const __restrict__ tjm) {
    // 22 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    const float dz = sz - tz;
    const float r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr2;
    const float invR = 1.0f / sqrtf(r2);
    const float invR2 = invR * invR;
    const float factor = sm * invR * invR2;
    *tax += factor * dx;
    *tay += factor * dy;
    *taz += factor * dz;
    *tjm += factor;
}

template <class T>
void __attribute__ ((noinline)) nbody_serial_accjerk(const int ns_start, const int ns_end,
                                     const T* const __restrict__ x,
                                     const T* const __restrict__ y,
                                     const T* const __restrict__ z,
                                     const T* const __restrict__ m,
                                     const T* const __restrict__ r,
                                     const int nt_start, const int nt_end,
                                     T* const __restrict__ ax,
                                     T* const __restrict__ ay,
                                     T* const __restrict__ az,
                                     T* const __restrict__ jm) {

    #pragma omp parallel for
    for (int i = nt_start; i < nt_end; i++) {
        const T tr2 = r[i]*r[i];
        for (int j = ns_start; j < ns_end; j++) {
            nbody_kernel_accjerk<T>(x[j], y[j], z[j], m[j], r[j],
                            x[i], y[i], z[i], tr2, &ax[i], &ay[i], &az[i], &jm[i]);
        }
    }
}

template <class T, class IT>
void nbody_serial_accjerk(const int ns_start, const int ns_end,
                  const VectorSoA<T,IT,3>& pos,
                  const std::vector<T>& m,
                  const std::vector<T>& r,
                  const int nt_start, const int nt_end,
                  VectorSoA<T,IT,3>& acc,
                  std::vector<T>& jerk) {

    nbody_serial_accjerk(ns_start, ns_end,
                 pos.x[0].data(), pos.x[1].data(), pos.x[2].data(),
                 m.data(), r.data(),
                 nt_start, nt_end,
                 acc.x[0].data(), acc.x[1].data(), acc.x[2].data(),
                 jerk.data());
}


//
// kernels to compute jerk magnitude on particles
//

// serial (x86) instructions
template <class T>
static inline void nbody_kernel_jerk(const T sx, const T sy, const T sz,
                                const T sm, const T sr,
                                const T tx, const T ty, const T tz, const T tr2,
                                T* const __restrict__ tjm) {
    // 17 flops
    const T dx = sx - tx;
    const T dy = sy - ty;
    const T dz = sz - tz;
    const T r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr2;
    const T invR = 1.0 / sqrt(r2);
    const T invR2 = invR * invR;
    *tjm += sm * invR * invR2;
}
// specialize for float
template <>
inline void nbody_kernel_jerk(const float sx, const float sy, const float sz,
                         const float sm, const float sr,
                         const float tx, const float ty, const float tz, const float tr2,
                         float* const __restrict__ tjm) {
    // 17 flops
    const float dx = sx - tx;
    const float dy = sy - ty;
    const float dz = sz - tz;
    const float r2 = dx*dx + dy*dy + dz*dz + sr*sr + tr2;
    const float invR = 1.0f / sqrtf(r2);
    const float invR2 = invR * invR;
    *tjm += sm * invR * invR2;
}

template <class T>
void __attribute__ ((noinline)) nbody_serial_jerk(const int ns_start, const int ns_end,
                                     const T* const __restrict__ x,
                                     const T* const __restrict__ y,
                                     const T* const __restrict__ z,
                                     const T* const __restrict__ m,
                                     const T* const __restrict__ r,
                                     const int nt_start, const int nt_end,
                                     T* const __restrict__ jm) {

    #pragma omp parallel for
    for (int i = nt_start; i < nt_end; i++) {
        const T tr2 = r[i]*r[i];
        for (int j = ns_start; j < ns_end; j++) {
            nbody_kernel_jerk<T>(x[j], y[j], z[j], m[j], r[j],
                            x[i], y[i], z[i], tr2, &jm[i]);
        }
    }
}

template <class T, class IT>
void nbody_serial_jerk(const int ns_start, const int ns_end,
                  const VectorSoA<T,IT,3>& pos,
                  const std::vector<T>& m,
                  const std::vector<T>& r,
                  const int nt_start, const int nt_end,
                  std::vector<T>& jerk) {

    nbody_serial_jerk(ns_start, ns_end,
                 pos.x[0].data(), pos.x[1].data(), pos.x[2].data(),
                 m.data(), r.data(),
                 nt_start, nt_end,
                 jerk.data());
}

