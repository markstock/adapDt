/*
 * ParticleSys.h - generic system of interacting particles
 *
 * Part of adapDt - testing hierarchical time stepping for gravitation simulations
 *
 * Copyright (c) 2023-4 Mark J Stock <markjstock@gmail.com>
*/
#pragma once

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
#include <cassert>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "State.h"
#include "GravKernels.h"


// Class to store particles in flat arrays
template <class T>
struct Particles {

  int n = 0;
  std::vector<int> idx;

  // this goes into the specific particle system header
  std::vector<T> r, m;
  // one State stores position, velocity, acceleration for n particles
  AccelerationState<T,int,3> s;

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
    const int nrings = (int)std::sqrt((T)n);
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
        s.pos.x[0][pc] = rad * std::cos(theta);
        s.pos.x[1][pc] = rad * std::sin(theta);
        s.pos.x[2][pc] = 0.0;
        ++pc;
      }
    }

    // and all vels and accs are zero
    s.vel.zero();
    s.acc.zero();

    // now compute accelerations on all particles
    nbody_serial<T>(0, n, s.pos, m, r, 0, n, s.acc);

    // set velocities to counteract inward acceleration (v^2 / r)
    std::uniform_real_distribution<> zbobble(-0.01, 0.01);
    for (int i=0; i<n; ++i) {
      const T rad = std::sqrt(std::pow(s.pos.x[0][i],2) + std::pow(s.pos.x[1][i],2) + std::pow(s.pos.x[2][i],2));
      const T accmag = std::sqrt(std::pow(s.acc.x[0][i],2) + std::pow(s.acc.x[1][i],2) + std::pow(s.acc.x[2][i],2));
      const T velmag = std::sqrt(accmag*rad);
      //std::cout << "part " << i << " at r=" << rad << " has accel " << accmag << " so vel is " << velmag << "\n";
      // bobble the z coordinate a little
      s.pos.x[2][i] = zbobble(_gen);
      // set the planar velocities now
      s.vel.x[0][i] = velmag * (-s.pos.x[1][i]/rad);
      s.vel.x[1][i] = velmag * (s.pos.x[0][i]/rad);
    }
  }

  void deep_copy(const Particles& _from) {
    assert(n == _from.n && "Copying from Particles with different n");
    for (int i=0; i<n; ++i) idx[i] = _from.idx[i];
    for (int i=0; i<n; ++i) r[i] = _from.r[i];
    for (int i=0; i<n; ++i) m[i] = _from.m[i];
  }

  void shallow_copy(const Particles& _from) {
    // warning: this can orphan memory!!!
    // need to implement this for vectors
    //n = _s.n;
    //r = _s.r;
    //m = _s.m;
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
      OUTFILE << idx[i] << " " << std::setprecision(17) << s.pos.x[0][i] << " " << s.pos.x[1][i] << " " << s.pos.x[2][i] << " " << r[i] << " " << m[i] << " " << s.vel.x[0][i] << " " << s.vel.x[1][i] << " " << s.vel.x[2][i] << "\n";
    }
    OUTFILE.close();
    std::cout << "done" << std::endl;
  }

  void fromfile(const std::string _ifn) {
    assert(not _ifn.empty() && "Input name is bad or empty");
    std::cout << "\nReading data from (" << _ifn << ")..." << std::flush;
    std::ifstream INFILE(_ifn);
    for (int i=0; i<n; ++i) {
      INFILE >> idx[i] >> s.pos.x[0][i] >> s.pos.x[1][i] >> s.pos.x[2][i] >> r[i] >> m[i] >> s.vel.x[0][i] >> s.vel.x[1][i] >> s.vel.x[2][i];
    }
    INFILE.close();
    std::cout << "done" << std::endl;
  }

  //
  // simple step: all source particles affect given target particles
  //
  void take_step (const double _dt, const int _trgstart, const int _trgend) {
    // zero the accelerations
    s.zero(_trgstart, _trgend);

    // run the O(N^2) force summation - all sources on the given targets
    nbody_serial(0, n, s.pos, m, r, _trgstart, _trgend, s.acc);

    // advect the particles
    s.euler_step(_dt, _trgstart, _trgend);
  }

  //
  // final step: given source particles affect given target particles, starting with given acceleration
  //
  void take_step (const double _dt, const int _trgstart, const int _trgend,
                  const VectorSoA<T,int,3>& _tempacc) {

    // set the accelerations from input (assume it's from all particles 0.._trgstart
    s.acc.set_from(_trgstart, _trgend, _tempacc, 0, _trgend-_trgstart);

    // run the O(N^2) force summation - all sources on the given targets
    nbody_serial(_trgstart, _trgend, s.pos, m, r, _trgstart, _trgend, s.acc);

    // advect the particles
    s.euler_step(_dt, _trgstart, _trgend);
  }

  //
  // a step where we split a range of targets into subranges of slow and fast particles,
  //   running one step of the slow particles and two steps of the fast particles
  // assumes particles with greatest error are nearer the end of the arrays
  //
  void take_step (const double _dt, const int _trgstart, const int _trgend,
                  const VectorSoA<T,int,3>& _tempacc, const float _slowfrac, const int _levels) {

    // where are we?
    static int max_levels = _levels;
    //for (int i=max_levels-_levels; i>0; --i) std::cout << "  ";
    //std::cout << "  stepping " << _trgstart << " to " << _trgend-1 << " by " << _dt << " at level " << _levels << std::endl;
    //std::cout << "  stepping " << _trgstart << " to " << _trgend-1 << " by " << _dt << " at level " << _levels << " with temp " << _tempacc.x[9999] << " saved " << s.acc.x[9999] << std::endl;

    // if there are no more levels to go, run the easy one
    if (_levels == 1) {
        take_step(_dt, _trgstart, _trgend, _tempacc);
        return;
    }

    // we're doing at least a 2-level time stepping
    assert(_slowfrac > 0.0 and _slowfrac < 1.0 and "_slowfrac is not 0..1");

    // find cutoff between slow (low index) and fast (higher index) particles
    const int ifast = _trgstart + (_trgend-_trgstart)*_slowfrac;

    // let's work on this step's slow particles first

    // set accelerations of slow target particles with passed-in values (all slower)
    s.acc.set_from(_trgstart, ifast, _tempacc, 0, ifast-_trgstart);

    // find accelerations of slow particles from all particles
    nbody_serial(_trgstart, _trgend, s.pos, m, r, _trgstart, ifast, s.acc);

    // now focus on the fast particles

    // set accelerations of fast target particles with passed-in values (all slower)
    s.acc.set_from(ifast, _trgend, _tempacc, ifast-_trgstart, _trgend-_trgstart);
    //std::cout << "           A tempacc " << _tempacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // find accelerations of fast particles from slow particles (using original positions of slow parts)
    nbody_serial(_trgstart, ifast, s.pos, m, r, ifast, _trgend, s.acc);
    //std::cout << "           B tempacc " << _tempacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // save the accelerations on all fast particles from all slow particles
    VectorSoA<T,int,3> slowacc(_trgend-ifast);
    // then from the slow portion of particles within this step
    slowacc.set_from(0, _trgend-ifast, s.acc, ifast, _trgend);
    // now slowacc holds the accelerations on all fast particles from all slower particles
    //std::cout << "           C slowacc " << slowacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // all fast particles take a half step
    take_step(0.5*_dt, ifast, _trgend, slowacc, _slowfrac, _levels-1);
    //std::cout << "           D slowacc " << slowacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // all fast particles take a half step again
    take_step(0.5*_dt, ifast, _trgend, slowacc, _slowfrac, _levels-1);
    //std::cout << "           E slowacc " << slowacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // before returning, move the slow particles
    s.euler_step(_dt, _trgstart, ifast);
  }

  //
  // a step where we split a range of targets into subranges of slow and fast particles,
  //   running one step of the slow particles and two steps of the fast particles
  // will re-sort all particles into slow and fast based on calculation of jerk magnitude
  //
  void take_step (const double _dt, const int _trgstart, const int _trgend,
                  const VectorSoA<T,int,3>& _tempacc, const std::vector<T>& _tempjerk, const int _levels) {

    // where are we?
    static int max_levels = _levels;
    //for (int i=max_levels-_levels; i>0; --i) std::cout << "  ";
    //std::cout << "  stepping " << _trgstart << " to " << _trgend-1 << " by " << _dt << " at level " << _levels << std::endl;
    //std::cout << "  stepping " << _trgstart << " to " << _trgend-1 << " by " << _dt << " at level " << _levels << " with temp " << _tempacc.x[9999] << " saved " << s.acc.x[9999] << std::endl;

    // if there are no more levels to go, run the easy one
    if (_levels == 1) {
        take_step(_dt, _trgstart, _trgend, _tempacc);
        return;
    }

    // how do we split slow and fast particles? magnitude of jerk
    // solve for accelerations and jerk in this step
    //nbody_serial_accjerk(_trgstart, _trgend, s.pos, m, r, _trgstart, _trgend, s.acc, s.jerk);

    // using jerk from last saved step, (re-)partition particles into slow and fast

    // find minimum jerk
    //std::vector<T>::iterator minidx = std::min_element(s.jerk.begin()+_trgstart, s.jerk.begin()+_trgend);
    auto minidx = std::min_element(s.jerk.begin()+_trgstart, s.jerk.begin()+_trgend);
    const T minjerk = *minidx;
    auto maxidx = std::max_element(s.jerk.begin()+_trgstart, s.jerk.begin()+_trgend);
    const T maxjerk = *maxidx;

    // double that to find the jerk value which separates fast from slow particles
    const T jerkpivot = 2.0 * minjerk;

    // now how many particles are slow?
    const int nslow = std::count_if(s.jerk.begin()+_trgstart, s.jerk.begin()+_trgend, [jerkpivot](T tj) { return tj < jerkpivot; });

    for (int i=max_levels-_levels; i>0; --i) std::cout << "  ";
    std::cout << "  level " << _levels << " with dt " << _dt << " has parts " << _trgstart << " to " << _trgend-1 << " with jerk " << minjerk << " to " << maxjerk << ", pivot at " << jerkpivot << " gives " << nslow << " slow parts" << std::endl;

    // if nslow is all parts, just take the one step
    if (nslow == _trgend-_trgstart) {
        take_step(_dt, _trgstart, _trgend, _tempacc);
        return;
    }

    // if we get here, we're doing at least a 2-level time stepping

    // find cutoff between slow (low index) and fast (higher index) particles
    const int ifast = _trgstart + nslow;

    // let's work on this step's slow particles first

    // set accelerations of slow target particles with passed-in values (all slower)
    s.acc.set_from(_trgstart, ifast, _tempacc, 0, ifast-_trgstart);

    // find accelerations and new jerks on slow particles from all particles
    nbody_serial_accjerk(_trgstart, _trgend, s.pos, m, r, _trgstart, ifast, s.acc, s.jerk);

    // now focus on the fast particles

    // set accelerations of fast target particles with passed-in values (all slower)
    s.acc.set_from(ifast, _trgend, _tempacc, ifast-_trgstart, _trgend-_trgstart);
    //std::cout << "           A tempacc " << _tempacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // find accelerations of fast particles from slow particles (using original positions of slow parts)
    nbody_serial_accjerk(_trgstart, ifast, s.pos, m, r, ifast, _trgend, s.acc, s.jerk);
    //std::cout << "           B tempacc " << _tempacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // save the accelerations on all fast particles from all slow particles
    VectorSoA<T,int,3> slowacc(_trgend-ifast);
    // then from the slow portion of particles within this step
    slowacc.set_from(0, _trgend-ifast, s.acc, ifast, _trgend);
    // now slowacc holds the accelerations on all fast particles from all slower particles
    //std::cout << "           C slowacc " << slowacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // all fast particles take a half step
    take_step(0.5*_dt, ifast, _trgend, slowacc, _tempjerk, _levels-1);
    //std::cout << "           D slowacc " << slowacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // all fast particles take a half step again
    take_step(0.5*_dt, ifast, _trgend, slowacc, _tempjerk, _levels-1);
    //std::cout << "           E slowacc " << slowacc.x[9999] << " saved " << s.acc.x[9999] << " at lev " << _levels << std::endl;

    // before returning, move the slow particles
    s.euler_step(_dt, _trgstart, ifast);
  }

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
  copy_state(_from.s, _to.s);
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
}
#endif

