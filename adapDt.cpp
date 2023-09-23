/*
 * adapDt - testing hierarchical time stepping for gravitation simulations
 *
 * originally ngrav3d.cpp from nvortexVc
 * Copyright (c) 2023 Mark J Stock <markjstock@gmail.com>
*/

#include <iostream>
#include <fstream>
#include <iomanip>		// for setprecision
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
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

// not really alignment, just minimum block sizes
inline int buffer(const int _n, const int _align) {
  // 63,64 returns 1*64; 64,64 returns 1*64; 65,64 returns 2*64=128
  return _align*((_n+_align-1)/_align);
}

// distribute ntotal items across nproc processes, now how many are on proc iproc?
inline int nthisproc(const int ntotal, const int iproc, const int nproc) {
  const int base_nper = ntotal / nproc;
  return (iproc < (ntotal-base_nper*nproc)) ? base_nper+1 : base_nper;
}

// Class to store a 3-vec for each particle in SoA style
template <class T>
struct ThreeVec {
  std::vector<T> x, y, z;

  // constructor needs a size
  ThreeVec(const int _n) {
    x.resize(_n);
    y.resize(_n);
    z.resize(_n);
    zero();
  }

  // destructor
  ~ThreeVec() { }

  void init_rand(std::mt19937 _gen, const T _min, const T _max) {
    std::uniform_real_distribution<> zmean_dist(_min, _max);
    const int n = x.size();
    for (int i=0; i<n; ++i) x[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) y[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) z[i] = zmean_dist(_gen);
  }

  void zero(const int _is, const int _if) {
    for (int i=_is; i<_if; ++i) x[i] = 0.0;
    for (int i=_is; i<_if; ++i) y[i] = 0.0;
    for (int i=_is; i<_if; ++i) z[i] = 0.0;
  }

  void zero() {
    zero(0, x.size());
  }

  void set_from(const int _is, const int _if, const ThreeVec<T>& _val) {
    for (int i=_is; i<_if; ++i) x[i] = _val.x[i];
    for (int i=_is; i<_if; ++i) y[i] = _val.y[i];
    for (int i=_is; i<_if; ++i) z[i] = _val.z[i];
  }

  void add_from(const int _is, const int _if, const ThreeVec<T>& _val) {
    for (int i=_is; i<_if; ++i) x[i] += _val.x[i];
    for (int i=_is; i<_if; ++i) y[i] += _val.y[i];
    for (int i=_is; i<_if; ++i) z[i] += _val.z[i];
  }

  void add_from(const int _is, const int _if, const std::vector<T>& _x, const std::vector<T>& _y, const std::vector<T>& _z) {
    for (int i=_is; i<_if; ++i) x[i] += _x[i];
    for (int i=_is; i<_if; ++i) y[i] += _y[i];
    for (int i=_is; i<_if; ++i) z[i] += _z[i];
  }

  void copy_to(const int _is, const int _if, std::vector<T>& _x, std::vector<T>& _y, std::vector<T>& _z) {
    for (int i=_is; i<_if; ++i) _x[i] = x[i];
    for (int i=_is; i<_if; ++i) _y[i] = y[i];
    for (int i=_is; i<_if; ++i) _z[i] = z[i];
  }

  void reorder(const std::vector<int>& _idx) {
    assert(_idx.size() == x.size() && "Idx and X vectors do not match in size");
    std::vector<T> tmp(_idx.size());
    const int n = x.size();
    std::copy(x.begin(), x.end(), tmp.begin());
    for (int i=0; i<n; ++i) x[i] = tmp[_idx[i]];
    std::copy(y.begin(), y.end(), tmp.begin());
    for (int i=0; i<n; ++i) y[i] = tmp[_idx[i]];
    std::copy(z.begin(), z.end(), tmp.begin());
    for (int i=0; i<n; ++i) z[i] = tmp[_idx[i]];
  }
};

// Class to store states
template <class T>
struct State {
  int n;
  ThreeVec<T> pos, vel, acc;
  std::vector<T> jerk;

  // constructor needs a size
  State(const int _n) : n(_n), pos(_n), vel(_n), acc(_n), jerk(_n) {}

  // destructor
  ~State() { }

  void init_rand(std::mt19937 _gen) {
    pos.init_rand(_gen, -1.0, 1.0);
    vel.init_rand(_gen, -1.0, 1.0);
    acc.zero();
  }

  void zero(const int _is, const int _if) {
    acc.zero(_is,_if);
    for (int i=_is; i<_if; ++i) jerk[i] = 0.0;
  }

  void zero() {
    zero(0, n);
  }

  void setacc(const int _is, const int _if, const ThreeVec<T>& _acc) {
    acc.set_from(_is, _if, _acc);
  }

  void reorder(const std::vector<int>& _idx) {
    pos.reorder(_idx);
    vel.reorder(_idx);
    acc.reorder(_idx);
    std::vector<T> tmp(_idx.size());
    std::copy(jerk.begin(), jerk.end(), tmp.begin());
    for (int i=0; i<n; ++i) jerk[i] = tmp[_idx[i]];
  }

  void euler_step(const double _dt, const int _ifirst, const int _ilast) {
    // really first order in velocity, midpoint rule for position
    for (int i=_ifirst; i<_ilast; ++i) pos.x[i] += _dt*(vel.x[i] + 0.5*_dt*acc.x[i]);
    for (int i=_ifirst; i<_ilast; ++i) pos.y[i] += _dt*(vel.y[i] + 0.5*_dt*acc.y[i]);
    for (int i=_ifirst; i<_ilast; ++i) pos.z[i] += _dt*(vel.z[i] + 0.5*_dt*acc.z[i]);
    for (int i=_ifirst; i<_ilast; ++i) vel.x[i] += _dt*acc.x[i];
    for (int i=_ifirst; i<_ilast; ++i) vel.y[i] += _dt*acc.y[i];
    for (int i=_ifirst; i<_ilast; ++i) vel.z[i] += _dt*acc.z[i];
  }
};

// Class to store particles in flat arrays
template <class T>
struct Particles {

  int n = 0;
  std::vector<int> idx;
  std::vector<T> r, m;
  // one State stores position, velocity, acceleration for n particles
  State<T> s;

  // sized constructor is the only constructor
  Particles(const int _n) : n(_n), s(_n) {
    idx.resize(_n);
    r.resize(_n);
    m.resize(_n);
    std::iota(idx.begin(), idx.end(), 0);
  }

  // destructor
  ~Particles() {}

  // copy constructor (must have same template parameter)
  Particles(const Particles& _src) : Particles(_src.n) {
    deep_copy(_src);
  }

  void init_rand(std::mt19937 _gen) {
    std::uniform_real_distribution<> zmean_mass(0.0, 1.0);
    for (int i=0; i<n; ++i) m[i] = zmean_mass(_gen);
    const T rad = 1.0 / std::sqrt((T)n);
    for (int i=0; i<n; ++i) r[i] = rad;
    s.init_rand(_gen);
  }

  void init_disk(std::mt19937 _gen) {
    // multiple rings extending to r=1 from origin, in xy plane (z small)
    // how many rings?
    const int nrings = (int)sqrt((T)n);
    // how much mass and area per ring?
    std::vector<T> mpr(nrings);
    T msum = 0.0;
    T asum = 0.0;
    for (int i=0; i<nrings; ++i) {
      mpr[i] = 1.0 / (i+0.5);
      msum += mpr[i];
      asum += (i+0.5);
    }
    for (int i=0; i<nrings; ++i) mpr[i] /= msum;
    //std::cout << "n is " << n << " and asum is " << asum << "\n";
    // loop over rings, adding particles as you go
    int nremain = n;
    int pc = 0;
    const T twopi = 2.0 * 3.14159265358979;
    std::uniform_real_distribution<> thetashift(0.0, 1.0);
    for (int i=0; i<nrings; ++i) {
      int nparts = std::max(2, std::min(nremain, (int)(0.5 + n*(i+0.5)/asum)));
      if (i==nrings-1) nparts = nremain;
      const T rad = 1.0 * (i+0.5) / nrings;
      std::cout << "ring " << i << " at rad " << rad << " has mass " << mpr[i] << " and " << nparts << " particles\n";
      nremain -= nparts;
      const T tshift = thetashift(_gen);
      for (int j=0; j<nparts; ++j) {
        m[pc] = mpr[i] / nparts;
        r[pc] = 0.25 / nrings;
        const T theta = twopi*(j+tshift)/nparts;
        s.pos.x[pc] = rad * std::cos(theta);
        s.pos.y[pc] = rad * std::sin(theta);
        s.pos.z[pc] = 0.0;
        ++pc;
      }
    }

    // and all vels and accs are zero
    s.vel.zero();
    s.acc.zero();

    // now compute accelerations on all particles
    nbody_serial<T>(0, n, s.pos.x.data(), s.pos.y.data(), s.pos.z.data(), m.data(), r.data(),
                    0, n, s.acc.x.data(), s.acc.y.data(), s.acc.z.data());

    // set velocities to counteract inward acceleration (v^2 / r)
    std::uniform_real_distribution<> zbobble(-0.01, 0.01);
    for (int i=0; i<n; ++i) {
      const T rad = std::sqrt(std::pow(s.pos.x[i],2) + std::pow(s.pos.y[i],2) + std::pow(s.pos.z[i],2));
      const T accmag = std::sqrt(std::pow(s.acc.x[i],2) + std::pow(s.acc.y[i],2) + std::pow(s.acc.z[i],2));
      const T velmag = std::sqrt(accmag*rad);
      //std::cout << "part " << i << " at r=" << rad << " has accel " << accmag << " so vel is " << velmag << "\n";
      // bobble the z coordinate a little
      s.pos.z[i] = zbobble(_gen);
      // set the planar velocities now
      s.vel.x[i] = velmag * (-s.pos.y[i]/rad);
      s.vel.y[i] = velmag * (s.pos.x[i]/rad);
    }
  }

  void deep_copy(const Particles& _from) {
    assert(n == _from.n && "Copying from Particles with different n");
    for (int i=0; i<n; ++i) idx[i] = _from.idx[i];
    for (int i=0; i<n; ++i) r[i] = _from.r[i];
    for (int i=0; i<n; ++i) s[i] = _from.s[i];
  }

  void shallow_copy(const Particles& _from) {
    // warning: this can orphan memory!!!
    // need to implement this for vectors
    //n = _s.n;
    //x = _s.x;
    //y = _s.y;
    //z = _s.z;
    //r = _s.r;
    //s = _s.s;
  }

  void reorder(const std::vector<int>& _idx) {
    // temporary reusable float vector
    std::vector<T> tmp(n);
    // copy the original input float vector into that temporary vector
    std::copy(r.begin(), r.end(), tmp.begin());
    // scatter values from the temp vector back into the original vector
    for (int i=0; i<n; ++i) r[i] = tmp[_idx[i]];

    // do it for everything else
    std::copy(idx.begin(), idx.end(), tmp.begin());
    for (int i=0; i<n; ++i) idx[i] = tmp[_idx[i]];
    std::copy(m.begin(), m.end(), tmp.begin());
    for (int i=0; i<n; ++i) m[i] = tmp[_idx[i]];
    s.reorder(_idx);
  }

  void tofile(const std::string _ofn) {
    assert(not _ofn.empty() && "Outfile name is bad or empty");
    std::cout << "\nWriting data to (" << _ofn << ")..." << std::flush;
    std::ofstream OUTFILE(_ofn);
    for (int i=0; i<n; ++i) {
      OUTFILE << idx[i] << " " << std::setprecision(17) << s.pos.x[i] << " " << s.pos.y[i] << " " << s.pos.z[i] << " " << r[i] << " " << m[i] << " " << s.vel.x[i] << " " << s.vel.y[i] << " " << s.vel.z[i] << "\n";
    }
    OUTFILE.close();
    std::cout << "done" << std::endl;
  }

  void fromfile(const std::string _ifn) {
    assert(not _ifn.empty() && "Input name is bad or empty");
    std::cout << "\nReading data from (" << _ifn << ")..." << std::flush;
    std::ifstream INFILE(_ifn);
    for (int i=0; i<n; ++i) {
      INFILE >> idx[i] >> s.pos.x[i] >> s.pos.y[i] >> s.pos.z[i] >> r[i] >> m[i] >> s.vel.x[i] >> s.vel.y[i] >> s.vel.z[i];
    }
    INFILE.close();
    std::cout << "done" << std::endl;
  }

  // simple step: all source particles affect given target particles
  void take_step (const double _dt, const int _trgstart, const int _trgend) {
    // zero the accelerations
    s.zero(_trgstart, _trgend);

    // run the O(N^2) force summation - all sources on the given targets
    nbody_serial(0, n, s.pos.x.data(), s.pos.y.data(), s.pos.z.data(), m.data(), r.data(),
                 _trgstart, _trgend, s.acc.x.data(), s.acc.y.data(), s.acc.z.data());

    // advect the particles
    s.euler_step(_dt, _trgstart, _trgend);
  }

  // final step: given source particles affect given target particles, starting with given acceleration
  void take_step (const double _dt, const int _trgstart, const int _trgend,
                  const ThreeVec<T>& _tempacc) {

    // set the accelerations from input (assume it's from all particles 0.._trgstart
    s.setacc(_trgstart, _trgend, _tempacc);

    // run the O(N^2) force summation - all sources on the given targets
    nbody_serial(_trgstart, _trgend, s.pos.x.data(), s.pos.y.data(), s.pos.z.data(), m.data(), r.data(),
                 _trgstart, _trgend, s.acc.x.data(), s.acc.y.data(), s.acc.z.data());

    // advect the particles
    s.euler_step(_dt, _trgstart, _trgend);
  }

  // a step where we split a range of targets into subranges of slow and fast particles,
  //   running one step of the slow particles and two steps of the fast particles
  // assumes particles with greatest error are nearer the end of the arrays
  //
  void take_step (const double _dt, const int _trgstart, const int _trgend,
                  const ThreeVec<T>& _tempacc, const float _slowfrac, const int _levels) {

    // where are we?
    static int max_levels = _levels;
    for (int i=max_levels-_levels; i>0; --i) std::cout << "  ";
    std::cout << "  stepping " << _trgstart << " to " << _trgend-1 << " by " << _dt << " at level " << _levels << std::endl;

    // if there are no more levels to go, run the easy one
    if (_levels == 1) {
        take_step(_dt, _trgstart, _trgend, _tempacc);
        return;
    }

    // we're doing at least a 2-level time stepping
    assert(_slowfrac > 0.0 and _slowfrac < 1.0 and "_slowfrac is not 0..1");

    // set accelerations of all target particles with passed-in values (all slower)
    s.setacc(_trgstart, _trgend, _tempacc);

    // find cutoff between slow (low index) and fast (higher index) particles
    const int ifast = _trgstart + (_trgend-_trgstart)*_slowfrac;

    // let's work on this step's slow particles first

    // find accelerations of slow particles from all particles
    nbody_serial(_trgstart, _trgend, s.pos.x.data(), s.pos.y.data(), s.pos.z.data(), m.data(), r.data(),
                 _trgstart, ifast, s.acc.x.data(), s.acc.y.data(), s.acc.z.data());

    // find accelerations of fast particles from slow particles
    nbody_serial(_trgstart, ifast, s.pos.x.data(), s.pos.y.data(), s.pos.z.data(), m.data(), r.data(),
                 ifast, _trgend, s.acc.x.data(), s.acc.y.data(), s.acc.z.data());

    // move the slow particles - we're done with them for this step
    s.euler_step(_dt, _trgstart, ifast);

    // now focus on the fast particles

    // save the accelerations on all fast particles from all slow particles
    //   note that these arrays are still size n - inefficient
    ThreeVec<T> slowacc(n);
    // then from the slow portion of particles within this step
    slowacc.add_from(ifast, _trgend, s.acc.x, s.acc.y, s.acc.z);

    // all fast particles take a half step
    take_step(0.5*_dt, ifast, _trgend, slowacc, _slowfrac, _levels-1);

    // all fast particles take a half step again
    take_step(0.5*_dt, ifast, _trgend, slowacc, _slowfrac, _levels-1);

  }
};

//
// Copy one Particles to another with different template param
//
template<class F, class T>
void copy_threevec (const ThreeVec<F>& _from, ThreeVec<T>& _to) {
  assert(_from.x.size() == _to.x.size() && "Copying from ThreeVec with different n");
  _to.x = std::vector<T>(_from.x.begin(), _from.x.end());
  _to.y = std::vector<T>(_from.y.begin(), _from.y.end());
  _to.z = std::vector<T>(_from.z.begin(), _from.z.end());
}
template<class F, class T>
void copy_particles (const Particles<F>& _from, Particles<T>& _to) {
  assert(_from.n == _to.n && "Copying from Particles with different n");
  _to.idx = std::vector<int>(_from.idx.begin(), _from.idx.end());
  _to.r = std::vector<T>(_from.r.begin(), _from.r.end());
  _to.m = std::vector<T>(_from.m.begin(), _from.m.end());
  copy_threevec(_from.s.pos, _to.s.pos);
  copy_threevec(_from.s.vel, _to.s.vel);
  copy_threevec(_from.s.acc, _to.s.acc);
}


#ifdef USE_MPI
// non-class method to swap sources
void __attribute__ ((noinline)) exchange_sources(const Particles& sendsrcs, const int ito,
                                                 Particles& recvsrcs, const int ifrom,
                                                 MPI_Request* handle) {

  // how many to send and receive?
  MPI_Sendrecv(&sendsrcs.n, 1, MPI_INT, ito, 9, &recvsrcs.n, 1, MPI_INT, ifrom, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //std::cout << "  proc " << iproc << " sending " << sendsrcs.n << " and receiving " << recvsrcs.n << std::endl;

  // do the transfers, saving the handle for future reference
  MPI_Irecv(recvsrcs.x, recvsrcs.n, MPI_FLOAT, ifrom, 10, MPI_COMM_WORLD, &handle[0]);
  MPI_Isend(sendsrcs.x, sendsrcs.n, MPI_FLOAT, ito, 10, MPI_COMM_WORLD, &handle[1]);
  MPI_Irecv(recvsrcs.y, recvsrcs.n, MPI_FLOAT, ifrom, 11, MPI_COMM_WORLD, &handle[2]);
  MPI_Isend(sendsrcs.y, sendsrcs.n, MPI_FLOAT, ito, 11, MPI_COMM_WORLD, &handle[3]);
  MPI_Irecv(recvsrcs.z, recvsrcs.n, MPI_FLOAT, ifrom, 12, MPI_COMM_WORLD, &handle[4]);
  MPI_Isend(sendsrcs.z, sendsrcs.n, MPI_FLOAT, ito, 12, MPI_COMM_WORLD, &handle[5]);
  MPI_Irecv(recvsrcs.r, recvsrcs.n, MPI_FLOAT, ifrom, 13, MPI_COMM_WORLD, &handle[6]);
  MPI_Isend(sendsrcs.r, sendsrcs.n, MPI_FLOAT, ito, 13, MPI_COMM_WORLD, &handle[7]);
  MPI_Irecv(recvsrcs.s, recvsrcs.n, MPI_FLOAT, ifrom, 14, MPI_COMM_WORLD, &handle[8]);
  MPI_Isend(sendsrcs.s, sendsrcs.n, MPI_FLOAT, ito, 14, MPI_COMM_WORLD, &handle[9]);
}
#endif

static inline void usage() {
    fprintf(stderr, "Usage: adapDt.bin [-n <number>] [-s <steps>] [-dt <num>] [-l <nlevels>] [-end <time>]\n");
    exit(1);
}

enum ParticleDistribution {
  cube,
  disk
};

// main program

int main(int argc, char *argv[]) {

    bool presort = true;
    bool runtrueonly = false;
    int num_levels = 1;
    int n_in = 10000;
    int nsteps = 1;
    int nproc = 1;
    int iproc = 0;
    double dt = 0.001;
    double endtime = 0.01;
    std::string infile;
    std::string outfile;
    std::string comparefile;
    std::string errorfile;
    ParticleDistribution distro = disk;

    for (int i=1; i<argc; i++) {
        if (strncmp(argv[i], "-n=", 3) == 0) {
            int num = atoi(argv[i] + 3);
            if (num < 1) usage();
            n_in = num;
        } else if (strncmp(argv[i], "-n", 2) == 0) {
            int num = atoi(argv[++i]);
            if (num < 1) usage();
            n_in = num;

        } else if (strncmp(argv[i], "-s=", 3) == 0) {
            int num = atoi(argv[i] + 3);
            if (num < 0) usage();
            nsteps = num;
        } else if (strncmp(argv[i], "-s", 2) == 0) {
            int num = atoi(argv[++i]);
            if (num < 0) usage();
            nsteps = num;

        } else if (strncmp(argv[i], "-dt=", 4) == 0) {
            double num = atof(argv[i] + 4);
            if (num < 0.0) usage();
            dt = num;
            nsteps = 0.5 + endtime / dt;
        } else if (strncmp(argv[i], "-dt", 3) == 0) {
            int num = atof(argv[++i]);
            if (num < 0.0) usage();
            dt = num;
            nsteps = 0.5 + endtime / dt;

        } else if (strncmp(argv[i], "-l=", 3) == 0) {
            int num = atoi(argv[i] + 3);
            if (num < 1) usage();
            num_levels = num;
        } else if (strncmp(argv[i], "-l", 2) == 0) {
            int num = atoi(argv[++i]);
            if (num < 1) usage();
            num_levels = num;

        } else if (strncmp(argv[i], "-end=", 5) == 0) {
            double num = atof(argv[i] + 5);
            if (num < 0.0) usage();
            endtime = num;
            nsteps = 0.5 + endtime / dt;
        } else if (strncmp(argv[i], "-end", 4) == 0) {
            int num = atof(argv[++i]);
            if (num < 0.0) usage();
            endtime = num;
            nsteps = 0.5 + endtime / dt;

        } else if (strncmp(argv[i], "-i=", 3) == 0) {
            infile = std::string(argv[i]);
            infile.erase(0,3);
        } else if (strncmp(argv[i], "-i", 2) == 0) {
            infile = std::string(argv[++i]);

        } else if (strncmp(argv[i], "-o=", 3) == 0) {
            outfile = std::string(argv[i]);
            outfile.erase(0,3);
        } else if (strncmp(argv[i], "-o", 2) == 0) {
            outfile = std::string(argv[++i]);

        } else if (strncmp(argv[i], "-c=", 3) == 0) {
            comparefile = std::string(argv[i]);
            comparefile.erase(0,3);
        } else if (strncmp(argv[i], "-c", 2) == 0) {
            comparefile = std::string(argv[++i]);

        } else if (strncmp(argv[i], "-error=", 7) == 0) {
            errorfile = std::string(argv[i]);
            errorfile.erase(0,7);
        } else if (strncmp(argv[i], "-error", 6) == 0) {
            errorfile = std::string(argv[++i]);

        } else if (strncmp(argv[i], "-true", 2) == 0) {
            runtrueonly = true;
        }
    }

#ifdef USE_MPI
    (void) MPI_Init(&argc, &argv);
    (void) MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    (void) MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    if (iproc==0) std::cout << "MPI-capable binary" << std::endl;

    // user specifies *total* number of particles, each processor take a share of those
    n_in = nthisproc(n_in, iproc, nproc);

    std::cout << "Proc " << iproc << " of " << nproc << " owns " << n_in << " particles" << std::endl;
#else
    std::cout << "Running with " << n_in << " particles" << std::endl;
#endif

    // set problem size
    const int maxGangSize = 1;		// this is 512 bits / 32 bits per float
    const int numSrcs = buffer(n_in,maxGangSize);

    // init random number generator
    //std::random_device rd;  //Will be used to obtain a seed for the random number engine
    //std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(12345); //Standard mersenne_twister_engine seeded with rd()

    // allocate fp64 and fp32 particle data
    Particles<double> orig(numSrcs);
    if (infile.empty()) {
        if (distro = cube) {
            // if random cloud and linear masses
            orig.init_rand(gen);
        } else {
            // if disk distribution with power law masses
            orig.init_disk(gen);
        }
    } else {
        orig.fromfile(infile);
    }

    // copy the full-precision particles to the test set
    Particles<float> src(numSrcs);
    copy_particles(orig, src);

    // optionally save initial state
    if (not outfile.empty() and nsteps == 0) {
        // we didn't run a sim, so save the initial state of the high-precision particles
        orig.tofile(outfile);
    }

#ifdef USE_MPI
    // what's the largest number that we'll encounter?
    int n_large = 0;
    MPI_Allreduce(&numSrcs, &n_large, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    // make two sets of arrays for communication
    Particles work = Particles(n_large);
    Particles buf = Particles(n_large);
    // and one set of pointers
    Particles ptr;

    // always pull from left (-), send to right (+)
    const int ifrom = (iproc == 0) ? nproc-1 : iproc-1;
    const int ito = (iproc == nproc-1) ? 0 : iproc+1;
    const int nreqs = 10;
    MPI_Request* xfers = new MPI_Request[nreqs];
    for (int ix=0; ix<nreqs; ++ix) xfers[ix] = MPI_REQUEST_NULL;
    //std::cout << "  proc " << iproc << " always sending to " << ito << " and receiving from " << ifrom << std::endl;
#endif

    //
    // optionally run one summation and then re-order the particles based on a criterion
    //
    if (presort) {
        printf("\nRunning precursor calculation and reordering\n");

        // find all accelerations
        orig.s.zero();
        nbody_serial_accjerk(0, orig.n, orig.s.pos.x.data(), orig.s.pos.y.data(), orig.s.pos.z.data(), orig.m.data(), orig.r.data(),
                     0, orig.n, orig.s.acc.x.data(), orig.s.acc.y.data(), orig.s.acc.z.data(), orig.s.jerk.data());

        // convert to a scalar value
        std::vector<double> val(orig.n);
        // if we use raw acceleration
        //for (int i=0; i<orig.n; ++i) val[i] = std::sqrt(orig.s.acc.x[i]*orig.s.acc.x[i] + orig.s.acc.y[i]*orig.s.acc.y[i] + orig.s.acc.z[i]*orig.s.acc.z[i]);
        // what if we use force as the orderer?
        //for (int i=0; i<orig.n; ++i) val[i] = orig.m[i] * std::sqrt(orig.s.acc.x[i]*orig.s.acc.x[i] + orig.s.acc.y[i]*orig.s.acc.y[i] + orig.s.acc.z[i]*orig.s.acc.z[i]);
        // and what about mass? the best for a single direct solve
        //for (int i=0; i<orig.n; ++i) val[i] = orig.m[i];
        // try jerk
        for (int i=0; i<orig.n; ++i) val[i] = orig.s.jerk[i];

        // sort, and keep the re-ordering
        std::vector<int> idx(orig.n);
        // initialize original index locations
        std::iota(idx.begin(), idx.end(), 0);
        printf("idx start %d %d %d and end %d %d %d\n", idx[0], idx[1], idx[2], idx[orig.n-3], idx[orig.n-2], idx[orig.n-1]);
        printf("val start %g %g %g and end %g %g %g\n", val[0], val[1], val[2], val[orig.n-3], val[orig.n-2], val[orig.n-1]);

        // finally sort, lowest to highest
        if (true) {
            // lowest to highest
            std::sort(idx.begin(),
                      idx.end(),
                      [&val](int i1, int i2) {return val[i1] < val[i2];});
        } else {
            // highest to lowest
            std::sort(idx.begin(),
                      idx.end(),
                      [&val](int i1, int i2) {return val[i1] > val[i2];});
        }
        printf("idx start %d %d %d and end %d %d %d\n", idx[0], idx[1], idx[2], idx[orig.n-3], idx[orig.n-2], idx[orig.n-1]);
        printf("val start %g %g %g and end %g %g %g\n", val[idx[0]], val[idx[1]], val[idx[2]], val[idx[orig.n-3]], val[idx[orig.n-2]], val[idx[orig.n-1]]);
        printf("median value is %g\n", val[idx[orig.n/2]]);

        // reorder all points
        orig.reorder(idx);
        copy_particles(orig, src);
    }

    //
    // Run the simulation a few time steps
    //
    if (not runtrueonly) {
    printf("\nRunning test solution for %d steps, dt %g\n", nsteps, dt);
    float tottime = 0.0;

    for (int istep = 0; istep < nsteps; ++istep) {
        printf("step\t%d\n", istep);
        auto start = std::chrono::steady_clock::now();

        // zero the accelerations
        src.s.zero();

#ifdef USE_MPI
        // first set to compute is self
        ptr.shallow_copy(src);

        for (int ibatch=0; ibatch<nproc ; ++ibatch) {
            // post non-blocking sends (from ptr) and receives (into buf)
            if (ibatch < nproc-1) {
                exchange_sources(ptr, ito, buf, ifrom, xfers);
            }

            // run the O(N^2) calculation concurrently using ptr for sources
            nbody_serial(0, ptr.n, ptr.x, ptr.y, ptr.z, ptr.s, ptr.r,
                         0, ptr.n, tax, tay, taz);

            // wait for new data to arrive before continuing
            if (ibatch < nproc-1) {
                // wait on all transfers to complete
                MPI_Waitall(nreqs, xfers, MPI_STATUS_IGNORE);
                // copy buf to work
                work.deep_copy(buf);
                // ptr now points at work
                ptr.shallow_copy(work);
            }
        }

        // and to ensure correct timings, wait for all MPI processes to finish
        MPI_Barrier(MPI_COMM_WORLD);
#else
        // save a running vector of the acceleration from slower particles
        ThreeVec<float> acc_from_slower(src.n);
        acc_from_slower.zero();

        // step all particles
        src.take_step(dt, 0, src.n, acc_from_slower, 0.5, num_levels);
#endif

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        if (iproc==0) printf("  test eval:\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        tottime += (float)elapsed_seconds.count();
    } // end for istep
    if (iproc==0) printf("\nTotal test time :\t[%.6f] seconds\n", tottime);
    } // end if 

    // get the true solution somehow
    if (comparefile.empty()) {
        printf("\nRunning 'true' solution");

        // run the "true" solution some number of steps
        double truedt = (runtrueonly ? dt : 0.00001);
        int truensteps = 0.5 + endtime / truedt;
        for (int istep = 0; istep < truensteps; ++istep) {
            std::cout << "." << std::flush;
            orig.take_step(truedt, 0, orig.n);
        }
        std::cout << std::endl;
    } else {
        // just read in the compare file
        orig.fromfile(comparefile);
    }

    if (iproc==0 and false) {
        printf("\nat end (t=%g), some positions\n", nsteps*dt);

        // write positions
        for (int i=src.n/10; i<src.n; i+=src.n/5) printf("   particle %d  fp32 %10.6f %10.6f %10.6f  fp64 %10.6f %10.6f %10.6f\n",i,src.s.pos.x[i],src.s.pos.y[i],src.s.pos.z[i],orig.s.pos.x[i],orig.s.pos.y[i],orig.s.pos.z[i]);
        // write accelerations
        //for (int i=0; i<2; i++) printf("   particle %d  fp32 %10.4f %10.4f %10.4f  fp64 %10.4f %10.4f %10.4f\n",i,src.s.acc.x[i],src.s.acc.y[i],src.s.acc.z[i],orig.s.acc.x[i],orig.s.acc.y[i],orig.s.acc.z[i]);
        printf("\n");
    }

    // compute error
    if (not runtrueonly) {

        // before computing errors, sort the true particles
        std::vector<int> idx(orig.n);
        // initialize original index locations
        std::iota(idx.begin(), idx.end(), 0);
        // lowest to highest
        std::vector<int>& test = orig.idx;
        std::sort(idx.begin(),
                  idx.end(),
                  [&test](int i1, int i2) {return test[i1] < test[i2];});
        // reorder all points
        orig.reorder(idx);

        double numer = 0.0;
        double denom = 0.0;
        double maxerr = 0.0;

        // write file for error analysis (velocity)
        if (not errorfile.empty()) {
            for (int i=0; i<orig.n; ++i) {
                denom += std::pow(orig.s.vel.x[i],2) + std::pow(orig.s.vel.y[i],2) + std::pow(orig.s.vel.z[i],2);
            }
            std::cout << "\nWriting error to (" << errorfile << ")..." << std::flush;
            std::ofstream OUTFILE(errorfile);
            for (int i=0; i<orig.n; ++i) {
                const double posmag = std::sqrt(std::pow(src.s.pos.x[i],2) + std::pow(src.s.pos.y[i],2) + std::pow(src.s.pos.z[i],2));
                const double velmag = std::sqrt(std::pow(src.s.vel.x[i],2) + std::pow(src.s.vel.y[i],2) + std::pow(src.s.vel.z[i],2));
                const double accmag = std::sqrt(std::pow(src.s.acc.x[i],2) + std::pow(src.s.acc.y[i],2) + std::pow(src.s.acc.z[i],2));
                const int oid = src.idx[i];
                double velerr = std::pow(orig.s.vel.x[oid]-src.s.vel.x[i],2) + std::pow(orig.s.vel.y[oid]-src.s.vel.y[i],2) + std::pow(orig.s.vel.z[oid]-src.s.vel.z[i],2);
                velerr = std::sqrt(orig.n*velerr/denom);
                OUTFILE << i << " " << std::setprecision(8) << posmag << " " << velmag << " " << accmag << " " << src.r[i] << " " << src.m[i] << " " << velerr << "\n";
            }
            OUTFILE.close();
            std::cout << "done" << std::endl;
        }

        if (iproc==0) printf("\nFinal error analysis :\n");

        numer = 0.0;
        denom = 0.0;
        maxerr = 0.0;
        for (int i=0; i<orig.n; ++i) {
            const int oid = src.idx[i];
            const double thiserr = std::pow(orig.s.pos.x[oid]-src.s.pos.x[i],2) + std::pow(orig.s.pos.y[oid]-src.s.pos.y[i],2) + std::pow(orig.s.pos.z[oid]-src.s.pos.z[i],2);
            if (thiserr > maxerr) maxerr = thiserr;
            numer += thiserr;
            denom += std::pow(orig.s.pos.x[i],2) + std::pow(orig.s.pos.y[i],2) + std::pow(orig.s.pos.z[i],2);
        }
        if (iproc==0) printf("\t\t(%.3e / %.3e mean/max RMS error vs. dble, position)\n", std::sqrt(numer/denom), std::sqrt(orig.n*maxerr/denom));

        numer = 0.0;
        denom = 0.0;
        maxerr = 0.0;
        for (int i=0; i<orig.n; ++i) {
            const int oid = src.idx[i];
            const double thiserr = std::pow(orig.s.vel.x[oid]-src.s.vel.x[i],2) + std::pow(orig.s.vel.y[oid]-src.s.vel.y[i],2) + std::pow(orig.s.vel.z[oid]-src.s.vel.z[i],2);
            if (thiserr > maxerr) maxerr = thiserr;
            numer += thiserr;
            denom += std::pow(orig.s.vel.x[i],2) + std::pow(orig.s.vel.y[i],2) + std::pow(orig.s.vel.z[i],2);
        }
        if (iproc==0) printf("\t\t(%.3e / %.3e mean/max RMS error vs. dble, velocity)\n", std::sqrt(numer/denom), std::sqrt(orig.n*maxerr/denom));

    }

    // write output particles, true or test solution
    if (not outfile.empty()) {
        if (runtrueonly) {
            orig.tofile(outfile);
        } else {
            src.tofile(outfile);
        }
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
