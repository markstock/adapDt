/*
 * adapDt - testing hierarchical time stepping for gravitation simulations
 *
 * originally ngrav3d.cpp from nvortexVc
 * Copyright (c) 2023-4 Mark J Stock <markjstock@gmail.com>
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

#include "ParticleSys.h"


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


static inline void usage() {
    fprintf(stderr, "Usage: adapDt.bin [-n <number>] [-s <steps>] [-dt <num>] [-l <nlevels>] [-adapt] [-end <time>]\n");
    exit(1);
}

enum ParticleDistribution {
  cube,
  disk
};

// main program

int main(int argc, char *argv[]) {

    bool presort = true;
    bool adaptive = false;
    bool runtrueonly = false;
    int num_levels = 1;
    int n_in = 10000;
    int nsteps = 1;
    int nproc = 1;
    int iproc = 0;
    double dt = 0.001;
    double endtime = 0.01;
    float slowfrac = 0.5;
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

        } else if (strncmp(argv[i], "-frac=", 6) == 0) {
            double num = atof(argv[i] + 6);
            if (num < 0.0) usage();
            slowfrac = num;
        } else if (strncmp(argv[i], "-frac", 5) == 0) {
            int num = atof(argv[++i]);
            if (num < 0.0) usage();
            slowfrac = num;

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

        } else if (strncmp(argv[i], "-adapt", 2) == 0) {
            adaptive = true;
            presort = false;
        } else if (strncmp(argv[i], "-true", 2) == 0) {
            runtrueonly = true;
        } else if (strncmp(argv[i], "-h", 2) == 0) {
            usage();
        } else {
            usage();
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
        nbody_serial_accjerk(0, orig.n, orig.s.pos, orig.m, orig.r, 0, orig.n, orig.s.acc, orig.s.jerk);

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
    // optionally run one summation to see the jerk array
    //
    if (adaptive) {
        printf("\nCalculating jerk magnitude to prepare for first step\n");

        // just find jerks, not accelerations
        orig.s.zero_jerk();
        nbody_serial_jerk(0, orig.n, orig.s.pos, orig.m, orig.r, 0, orig.n, orig.s.jerk);
        // what did this look like?
        //printf("  jerk start %g %g %g and end %g %g %g\n", orig.s.jerk[0], orig.s.jerk[1], orig.s.jerk[2], orig.s.jerk[orig.n-3], orig.s.jerk[orig.n-2], orig.s.jerk[orig.n-1]);
        // make sure jerk copies over
        copy_particles(orig, src);
        //printf("  jerk start %g %g %g and end %g %g %g\n", src.s.jerk[0], src.s.jerk[1], src.s.jerk[2], src.s.jerk[src.n-3], src.s.jerk[src.n-2], src.s.jerk[src.n-1]);
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
        VectorSoA<float,int,3> acc_from_slower(src.n);
        acc_from_slower.zero();
        std::vector<float> jerk_from_slower(src.n);
        //jerk_from_slower.zero();

        // step all particles
        if (adaptive) {
            src.take_step(dt, 0, src.n, acc_from_slower, jerk_from_slower, num_levels);
        } else {
            src.take_step(dt, 0, src.n, acc_from_slower, slowfrac, num_levels);
        }
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
            //orig.take_step(truedt);
            orig.take_step_rk2(truedt);
        }
        std::cout << std::endl;
    } else {
        // just read in the compare file
        orig.fromfile(comparefile);
    }

    if (iproc==0 and false) {
        printf("\nat end (t=%g), some positions\n", nsteps*dt);

        // write positions
        for (int i=src.n/10; i<src.n; i+=src.n/5) printf("   particle %d  fp32 %10.6f %10.6f %10.6f  fp64 %10.6f %10.6f %10.6f\n",i,src.s.pos.x[0][i],src.s.pos.x[1][i],src.s.pos.x[2][i],orig.s.pos.x[0][i],orig.s.pos.x[1][i],orig.s.pos.x[2][i]);
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
                denom += std::pow(orig.s.vel.x[0][i],2) + std::pow(orig.s.vel.x[1][i],2) + std::pow(orig.s.vel.x[2][i],2);
            }
            std::cout << "\nWriting error to (" << errorfile << ")..." << std::flush;
            std::ofstream OUTFILE(errorfile);
            for (int i=0; i<orig.n; ++i) {
                const double posmag = std::sqrt(std::pow(src.s.pos.x[0][i],2) + std::pow(src.s.pos.x[1][i],2) + std::pow(src.s.pos.x[2][i],2));
                const double velmag = std::sqrt(std::pow(src.s.vel.x[0][i],2) + std::pow(src.s.vel.x[1][i],2) + std::pow(src.s.vel.x[2][i],2));
                const double accmag = std::sqrt(std::pow(src.s.acc.x[0][i],2) + std::pow(src.s.acc.x[1][i],2) + std::pow(src.s.acc.x[2][i],2));
                const int oid = src.idx[i];
                double velerr = std::pow(orig.s.vel.x[0][oid]-src.s.vel.x[0][i],2) + std::pow(orig.s.vel.x[1][oid]-src.s.vel.x[1][i],2) + std::pow(orig.s.vel.x[2][oid]-src.s.vel.x[2][i],2);
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
            const double thiserr = std::pow(orig.s.pos.x[0][oid]-src.s.pos.x[0][i],2) + std::pow(orig.s.pos.x[1][oid]-src.s.pos.x[1][i],2) + std::pow(orig.s.pos.x[2][oid]-src.s.pos.x[2][i],2);
            if (thiserr > maxerr) maxerr = thiserr;
            numer += thiserr;
            denom += std::pow(orig.s.pos.x[0][i],2) + std::pow(orig.s.pos.x[1][i],2) + std::pow(orig.s.pos.x[2][i],2);
        }
        if (iproc==0) printf("\t\t(%.3e / %.3e mean/max RMS error vs. dble, position)\n", std::sqrt(numer/denom), std::sqrt(orig.n*maxerr/denom));

        numer = 0.0;
        denom = 0.0;
        maxerr = 0.0;
        for (int i=0; i<orig.n; ++i) {
            const int oid = src.idx[i];
            const double thiserr = std::pow(orig.s.vel.x[0][oid]-src.s.vel.x[0][i],2) + std::pow(orig.s.vel.x[1][oid]-src.s.vel.x[1][i],2) + std::pow(orig.s.vel.x[2][oid]-src.s.vel.x[2][i],2);
            if (thiserr > maxerr) maxerr = thiserr;
            numer += thiserr;
            denom += std::pow(orig.s.vel.x[0][i],2) + std::pow(orig.s.vel.x[1][i],2) + std::pow(orig.s.vel.x[2][i],2);
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
