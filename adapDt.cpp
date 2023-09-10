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

// Class to store states
template <class T>
struct State {
  int n;
  std::vector<T> x, y, z;
  std::vector<T> vx, vy, vz;
  std::vector<T> ax, ay, az;

  // constructor needs a size
  State(const int _n) : n(_n) {
    x.resize(_n);
    y.resize(_n);
    z.resize(_n);
    vx.resize(_n);
    vy.resize(_n);
    vz.resize(_n);
    ax.resize(_n);
    ay.resize(_n);
    az.resize(_n);
  }

  // destructor
  ~State() { }

  void init_rand(std::mt19937 _gen) {
    std::uniform_real_distribution<> zmean_dist(-1.0, 1.0);
    for (int i=0; i<n; ++i) x[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) y[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) z[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) vx[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) vy[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) vz[i] = zmean_dist(_gen);
    for (int i=0; i<n; ++i) ax[i] = 0.0;
    for (int i=0; i<n; ++i) ay[i] = 0.0;
    for (int i=0; i<n; ++i) az[i] = 0.0;
  }

  void zero() {
    for (int i=0; i<n; ++i) ax[i] = 0.0;
    for (int i=0; i<n; ++i) ay[i] = 0.0;
    for (int i=0; i<n; ++i) az[i] = 0.0;
  }

  void reorder(const std::vector<int>& _idx) {
    // see the one in Particles
    std::vector<T> tmp(n);
    // positions
    std::copy(x.begin(), x.end(), tmp.begin());
    for (int i=0; i<n; ++i) x[i] = tmp[_idx[i]];
    std::copy(y.begin(), y.end(), tmp.begin());
    for (int i=0; i<n; ++i) y[i] = tmp[_idx[i]];
    std::copy(z.begin(), z.end(), tmp.begin());
    for (int i=0; i<n; ++i) z[i] = tmp[_idx[i]];
    // velocities
    std::copy(vx.begin(), vx.end(), tmp.begin());
    for (int i=0; i<n; ++i) vx[i] = tmp[_idx[i]];
    std::copy(vy.begin(), vy.end(), tmp.begin());
    for (int i=0; i<n; ++i) vy[i] = tmp[_idx[i]];
    std::copy(vz.begin(), vz.end(), tmp.begin());
    for (int i=0; i<n; ++i) vz[i] = tmp[_idx[i]];
    // don't bother with accelerations?
    std::copy(ax.begin(), ax.end(), tmp.begin());
    for (int i=0; i<n; ++i) ax[i] = tmp[_idx[i]];
    std::copy(ay.begin(), ay.end(), tmp.begin());
    for (int i=0; i<n; ++i) ay[i] = tmp[_idx[i]];
    std::copy(az.begin(), az.end(), tmp.begin());
    for (int i=0; i<n; ++i) az[i] = tmp[_idx[i]];
  }

  void euler_step(const double _dt, const int _ifirst, const int _ilast) {
    // really first order in velocity, midpoint rule for position
    for (int i=_ifirst; i<_ilast; ++i) x[i] += _dt*(vx[i] + 0.5*_dt*ax[i]);
    for (int i=_ifirst; i<_ilast; ++i) y[i] += _dt*(vy[i] + 0.5*_dt*ay[i]);
    for (int i=_ifirst; i<_ilast; ++i) z[i] += _dt*(vz[i] + 0.5*_dt*az[i]);
    for (int i=_ifirst; i<_ilast; ++i) vx[i] += _dt*ax[i];
    for (int i=_ifirst; i<_ilast; ++i) vy[i] += _dt*ay[i];
    for (int i=_ifirst; i<_ilast; ++i) vz[i] += _dt*az[i];
  }
};

// Class to store particles in flat arrays
template <class T>
struct Particles {

  int n = 0;
  std::vector<int> idx;
  std::vector<T> r, m;
  State<T> s;

  // sized constructor is the only constructor
  Particles(const int _n) : n(_n), s(_n) {
    idx.resize(_n);
    r.resize(_n);
    m.resize(_n);
    std::iota(idx.begin(), idx.end(), 0);
  }

  // copy constructor (must have same template parameter)
  Particles(const Particles& _src) : Particles(_src.n) {
    deep_copy(_src);
  }

  void init_rand(std::mt19937 _gen) {
    s.init_rand(_gen);
    std::uniform_real_distribution<> zmean_mass(0.0, 1.0);
    for (int i=0; i<n; ++i) m[i] = zmean_mass(_gen);
    const T rad = 1.0 / std::sqrt((T)n);
    for (int i=0; i<n; ++i) r[i] = rad;
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
      OUTFILE << idx[i] << " " << std::setprecision(17) << s.x[i] << " " << s.y[i] << " " << s.z[i] << " " << r[i] << " " << m[i] << " " << s.vx[i] << " " << s.vy[i] << " " << s.vz[i] << "\n";
    }
    OUTFILE.close();
    std::cout << "done" << std::endl;
  }

  void fromfile(const std::string _ifn) {
    assert(not _ifn.empty() && "Input name is bad or empty");
    std::cout << "\nReading data from (" << _ifn << ")..." << std::flush;
    std::ifstream INFILE(_ifn);
    for (int i=0; i<n; ++i) {
      INFILE >> idx[i] >> s.x[i] >> s.y[i] >> s.z[i] >> r[i] >> m[i] >> s.vx[i] >> s.vy[i] >> s.vz[i];
    }
    INFILE.close();
    std::cout << "done" << std::endl;
  }

  // destructor
  ~Particles() { }
};

//
// Copy one Particles to another with different template param
//
template<class F, class T>
void copy_particles (const Particles<F>& _from, Particles<T>& _to) {
  assert(_from.n == _to.n && "Copying from Particles with different n");
  _to.idx = std::vector<int>(_from.idx.begin(), _from.idx.end());
  _to.r = std::vector<T>(_from.r.begin(), _from.r.end());
  _to.m = std::vector<T>(_from.m.begin(), _from.m.end());
  _to.s.x = std::vector<T>(_from.s.x.begin(), _from.s.x.end());
  _to.s.y = std::vector<T>(_from.s.y.begin(), _from.s.y.end());
  _to.s.z = std::vector<T>(_from.s.z.begin(), _from.s.z.end());
  _to.s.vx = std::vector<T>(_from.s.vx.begin(), _from.s.vx.end());
  _to.s.vy = std::vector<T>(_from.s.vy.begin(), _from.s.vy.end());
  _to.s.vz = std::vector<T>(_from.s.vz.begin(), _from.s.vz.end());
  _to.s.ax = std::vector<T>(_from.s.ax.begin(), _from.s.ax.end());
  _to.s.ay = std::vector<T>(_from.s.ay.begin(), _from.s.ay.end());
  _to.s.az = std::vector<T>(_from.s.az.begin(), _from.s.az.end());
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
    fprintf(stderr, "Usage: adapDt.bin [-n <number>] [-s <steps>]\n");
    exit(1);
}

// main program

int main(int argc, char *argv[]) {

    bool presort = true;
    bool twolevel = false;
    bool runtrueonly = false;
    int n_in = 10000;
    int nsteps = 1;
    int nproc = 1;
    int iproc = 0;
    double dt = 0.001;
    double endtime = 0.01;
    std::string infile;
    std::string outfile;
    std::string comparefile;

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

        } else if (strncmp(argv[i], "-true", 2) == 0) {
            runtrueonly = true;
        }
    }

    // median acceleration scales linearly with N
    //dt = 100.0 / (double)n_in;

    double truedt = 0.00001;
    int truensteps = 0.5 + endtime / truedt;

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
        orig.init_rand(gen);
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
        nbody_serial(0, orig.n, orig.s.x.data(), orig.s.y.data(), orig.s.z.data(), orig.m.data(), orig.r.data(),
                     0, orig.n, orig.s.ax.data(), orig.s.ay.data(), orig.s.az.data());

        // convert to a scalar value
        std::vector<double> val(orig.n);
        // if we use raw acceleration
        //for (int i=0; i<orig.n; ++i) val[i] = std::sqrt(orig.s.ax[i]*orig.s.ax[i] + orig.s.ay[i]*orig.s.ay[i] + orig.s.az[i]*orig.s.az[i]);
        // what if we use force as the orderer?
        for (int i=0; i<orig.n; ++i) val[i] = orig.m[i] * std::sqrt(orig.s.ax[i]*orig.s.ax[i] + orig.s.ay[i]*orig.s.ay[i] + orig.s.az[i]*orig.s.az[i]);
        // and what about mass? the best for a single direct solve
        //for (int i=0; i<orig.n; ++i) val[i] = orig.m[i];

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
        if (twolevel) {
            // first half-step
            nbody_serial(0, src.n, src.s.x.data(), src.s.y.data(), src.s.z.data(), src.m.data(), src.r.data(),
                         0, src.n, src.s.ax.data(), src.s.ay.data(), src.s.az.data());
            src.s.euler_step(0.5*dt, 0, src.n);
            // second half-step
            src.s.zero();
            nbody_serial(0, src.n, src.s.x.data(), src.s.y.data(), src.s.z.data(), src.m.data(), src.r.data(),
                         0, src.n, src.s.ax.data(), src.s.ay.data(), src.s.az.data());
            src.s.euler_step(0.5*dt, 0, src.n);
        } else {
            // run the O(N^2) calculation
            nbody_serial(0, src.n, src.s.x.data(), src.s.y.data(), src.s.z.data(), src.m.data(), src.r.data(),
                         0, src.n, src.s.ax.data(), src.s.ay.data(), src.s.az.data());

            // advect the particles
            src.s.euler_step(dt, 0, src.n);
        }
#endif

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        if (iproc==0) printf("  test eval:\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());
        tottime += (float)elapsed_seconds.count();

        //
        // repeat for doubles - always step all particles as "fast"
        //
        if (false) {
            start = std::chrono::steady_clock::now();
            if (twolevel) {
                // first half-step
                orig.s.zero();
                nbody_serial(0, orig.n, orig.s.x.data(), orig.s.y.data(), orig.s.z.data(), orig.m.data(), orig.r.data(),
                             0, orig.n, orig.s.ax.data(), orig.s.ay.data(), orig.s.az.data());
                orig.s.euler_step(0.5*dt, 0, orig.n);
                // second half-step
                orig.s.zero();
                nbody_serial(0, orig.n, orig.s.x.data(), orig.s.y.data(), orig.s.z.data(), orig.m.data(), orig.r.data(),
                             0, orig.n, orig.s.ax.data(), orig.s.ay.data(), orig.s.az.data());
                orig.s.euler_step(0.5*dt, 0, orig.n);
            } else {
                // full step
                orig.s.zero();
                nbody_serial(0, orig.n, orig.s.x.data(), orig.s.y.data(), orig.s.z.data(), orig.m.data(), orig.r.data(),
                             0, orig.n, orig.s.ax.data(), orig.s.ay.data(), orig.s.az.data());
                orig.s.euler_step(dt, 0, orig.n);
            }
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            if (iproc==0) printf("  dble eval:\t\t[%.6f] seconds\n", (float)elapsed_seconds.count());

            // compute error
            double numer = 0.0;
            double denom = 0.0;
            for (int i=0; i<orig.n; ++i) {
                numer += std::pow(orig.s.ax[i]-src.s.ax[i],2) + std::pow(orig.s.ay[i]-src.s.ay[i],2) + std::pow(orig.s.az[i]-src.s.az[i],2);
                denom += std::pow(orig.s.ax[i],2) + std::pow(orig.s.ay[i],2) + std::pow(orig.s.az[i],2);
            }
            if (iproc==0) printf("\t\t\t(%.3e RMS error vs. dble, acceleration)\n", std::sqrt(numer/denom));

            numer = 0.0;
            denom = 0.0;
            for (int i=0; i<orig.n; ++i) {
                numer += std::pow(orig.s.vx[i]-src.s.vx[i],2) + std::pow(orig.s.vy[i]-src.s.vy[i],2) + std::pow(orig.s.vz[i]-src.s.vz[i],2);
                denom += std::pow(orig.s.vx[i],2) + std::pow(orig.s.vy[i],2) + std::pow(orig.s.vz[i],2);
            }
            if (iproc==0) printf("\t\t\t(%.3e RMS error vs. dble, velocity)\n", std::sqrt(numer/denom));

            numer = 0.0;
            denom = 0.0;
            for (int i=0; i<orig.n; ++i) {
                numer += std::pow(orig.s.x[i]-src.s.x[i],2) + std::pow(orig.s.y[i]-src.s.y[i],2) + std::pow(orig.s.z[i]-src.s.z[i],2);
                denom += std::pow(orig.s.x[i],2) + std::pow(orig.s.y[i],2) + std::pow(orig.s.z[i],2);
            }
            if (iproc==0) printf("\t\t\t(%.3e RMS error vs. dble, position)\n", std::sqrt(numer/denom));
        }
    }
        if (iproc==0) printf("\nTotal test time :\t[%.6f] seconds\n", tottime);
    }

    // get the true solution somehow
    if (comparefile.empty()) {
        printf("\nRunning 'true' solution");
        // run the "true" solution some number of steps
        for (int istep = 0; istep < truensteps; ++istep) {
            std::cout << "." << std::flush;
            orig.s.zero();
            nbody_serial(0, orig.n, orig.s.x.data(), orig.s.y.data(), orig.s.z.data(), orig.m.data(), orig.r.data(),
                         0, orig.n, orig.s.ax.data(), orig.s.ay.data(), orig.s.az.data());
            orig.s.euler_step(truedt, 0, orig.n);
        }
        std::cout << std::endl;
    } else {
        // just read in the compare file
        orig.fromfile(comparefile);
    }

    if (iproc==0) {
        printf("\nat end (t=%g), some positions\n", nsteps*dt);

        // write positions
        for (int i=src.n/10; i<src.n; i+=src.n/5) printf("   particle %d  fp32 %10.6f %10.6f %10.6f  fp64 %10.6f %10.6f %10.6f\n",i,src.s.x[i],src.s.y[i],src.s.z[i],orig.s.x[i],orig.s.y[i],orig.s.z[i]);
        // write accelerations
        //for (int i=0; i<2; i++) printf("   particle %d  fp32 %10.4f %10.4f %10.4f  fp64 %10.4f %10.4f %10.4f\n",i,src.s.ax[i],src.s.ay[i],src.s.az[i],orig.s.ax[i],orig.s.ay[i],orig.s.az[i]);
        printf("\n");
    }

    // compute error
    if (not runtrueonly) {
        double numer = 0.0;
        double denom = 0.0;
        double maxerr = 0.0;
        /*
        for (int i=0; i<orig.n; ++i) {
            numer += std::pow(orig.s.ax[i]-src.s.ax[i],2) + std::pow(orig.s.ay[i]-src.s.ay[i],2) + std::pow(orig.s.az[i]-src.s.az[i],2);
            denom += std::pow(orig.s.ax[i],2) + std::pow(orig.s.ay[i],2) + std::pow(orig.s.az[i],2);
        }
        if (iproc==0) printf("\t\t\t(%.3e RMS error vs. dble, acceleration)\n", std::sqrt(numer/denom));

        numer = 0.0;
        denom = 0.0;
        maxerr = 0.0;
        */
        for (int i=0; i<orig.n; ++i) {
            double thiserr = std::pow(orig.s.vx[i]-src.s.vx[i],2) + std::pow(orig.s.vy[i]-src.s.vy[i],2) + std::pow(orig.s.vz[i]-src.s.vz[i],2);
            if (thiserr > maxerr) maxerr = thiserr;
            numer += thiserr;
            denom += std::pow(orig.s.vx[i],2) + std::pow(orig.s.vy[i],2) + std::pow(orig.s.vz[i],2);
        }
        if (iproc==0) printf("\t\t\t(%.3e / %.3e mean/max RMS error vs. dble, velocity)\n", std::sqrt(numer/denom), std::sqrt(orig.n*maxerr/denom));

        numer = 0.0;
        denom = 0.0;
        maxerr = 0.0;
        for (int i=0; i<orig.n; ++i) {
            double thiserr = std::pow(orig.s.x[i]-src.s.x[i],2) + std::pow(orig.s.y[i]-src.s.y[i],2) + std::pow(orig.s.z[i]-src.s.z[i],2);
            if (thiserr > maxerr) maxerr = thiserr;
            numer += thiserr;
            denom += std::pow(orig.s.x[i],2) + std::pow(orig.s.y[i],2) + std::pow(orig.s.z[i],2);
        }
        if (iproc==0) printf("\t\t\t(%.3e / %.3e mean/max RMS error vs. dble, position)\n", std::sqrt(numer/denom), std::sqrt(orig.n*maxerr/denom));
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
