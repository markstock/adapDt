/*
 * State.h - header for system dynamical state
 *
 * Part of adapDt - testing hierarchical time stepping for gravitation simulations
 *
 * Copyright (c) 2023-4 Mark J Stock <markjstock@gmail.com>
*/
#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <cassert>

#include "VectorSoA.h"

//
// Class to store dynamical states for both velocity and acceleration systems
// T is float storage class
// IT is integer (index) storage class (typically uint32_t or size_t)
// D is number of spatial dimensions
//

//
// State for acceleration/force systems like spring-mass-damper networks, gravitation, electrodynamics
//
template <class T, class IT, uint8_t D>
struct AccelerationState {

  VectorSoA<T,IT,D> pos, vel, acc;
  std::vector<T> jerk;

  // constructor needs a size
  AccelerationState(const IT _n) : pos(_n), vel(_n), acc(_n), jerk(_n) {}

  // alternatively, construct from an existing object
  AccelerationState(const AccelerationState& _from) :
    pos(_from.pos), vel(_from.vel), acc(_from.acc), jerk(_from.jerk) {}

  // destructor
  ~AccelerationState() { }

  void resize(const int _n) {
    pos.resize(_n);
    vel.resize(_n);
    acc.resize(_n);
    jerk.resize(_n);
  }

  void init_rand(std::mt19937 _gen) {
    pos.init_rand(_gen, -1.0, 1.0);
    vel.init_rand(_gen, -1.0, 1.0);
    acc.zero();
  }

  void zero(const IT _is, const IT _if) {
    acc.zero(_is,_if);
  }

  void zero() {
    zero(0, (IT)pos.x[0].size());
  }

  void zero_jerk(const IT _is, const IT _if) {
    for (IT i=_is; i<_if; ++i) jerk[i] = 0.0;
  }

  void zero_jerk() {
    zero_jerk(0, (IT)jerk.size());
  }

  void reorder(const std::vector<IT>& _idx) {
    pos.reorder(_idx);
    vel.reorder(_idx);
    acc.reorder(_idx);
    std::vector<T> tmp(_idx.size());
    std::copy(jerk.begin(), jerk.end(), tmp.begin());
    for (IT i=0; i<jerk.size(); ++i) jerk[i] = tmp[_idx[i]];
  }

  // computes new position and vel from pos, vel, acc, dt
  void euler_step(const double _dt, const IT _ifirst, const IT _ilast) {
    // be general: Euler step for both pos and vel
    for (uint8_t d=0; d<D; ++d) {
      for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += _dt*vel.x[d][i];
      for (IT i=_ifirst; i<_ilast; ++i) vel.x[d][i] += _dt*acc.x[d][i];
    }

    // really first order in velocity, midpoint rule for position
    //for (uint8_t d=0; d<D; ++d) {
    //  for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += _dt*(vel.x[d][i] + 0.5*_dt*acc.x[d][i]);
    //  for (IT i=_ifirst; i<_ilast; ++i) vel.x[d][i] += _dt*acc.x[d][i];
    //}

    // could also do
    //for (uint8_t d=0; d<D; ++d) {
    //  for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += 0.5*_dt*vel.x[d][i];
    //  for (IT i=_ifirst; i<_ilast; ++i) vel.x[d][i] += _dt*acc.x[d][i];
    //  for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += 0.5*_dt*vel.x[d][i];
    //}

    // or even
    //for (uint8_t d=0; d<D; ++d) {
    //  for (IT i=_ifirst; i<_ilast; ++i) {
    //    pos.x[d][i] += 0.5*_dt*vel.x[d][i];
    //    vel.x[d][i] += _dt*acc.x[d][i];
    //    pos.x[d][i] += 0.5*_dt*vel.x[d][i];
    //  }
    //}
  }

  // computes new position and vel from pos, vel, acc, dt, using both accelerations
  void rk2_step(const double _dt, const IT _ifirst, const IT _ilast, const AccelerationState& _alt) {
    // second order in velocity and position
    const double hdt = 0.5*_dt;
    for (uint8_t d=0; d<D; ++d) {
      for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += hdt * (vel.x[d][i]+_alt.vel.x[d][i]);
      for (IT i=_ifirst; i<_ilast; ++i) vel.x[d][i] += hdt * (acc.x[d][i]+_alt.acc.x[d][i]);
    }
  }

  // computes new position and vel from pos, vel, acc, dt, using this and 3 other stages
  void rk4_step(const double _dt, const IT _ifirst, const IT _ilast, const AccelerationState& _s2,
      const AccelerationState& _s3, const AccelerationState& _s4) {
    const double os = _dt/6.;
    const double ot = _dt/3.;
    // fourth order in velocity and position
    for (uint8_t d=0; d<D; ++d) {
      for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += os*vel.x[d][i] + ot*_s2.vel.x[d][i] + ot*_s3.vel.x[d][i] + os*_s4.vel.x[d][i];
      for (IT i=_ifirst; i<_ilast; ++i) vel.x[d][i] += os*acc.x[d][i] + ot*_s2.acc.x[d][i] + ot*_s3.acc.x[d][i] + os*_s4.acc.x[d][i];
    }
  }
};


template<class F, class FIT, class T, class TIT, uint8_t D>
void copy_state (const AccelerationState<F,FIT,D>& _from, AccelerationState<T,TIT,D>& _to) {
  //fprintf(stderr,"  from has %d  to has %d  particles\n", _from.jerk.size(), _to.jerk.size());
  assert(_from.jerk.size() == _to.jerk.size() && "Copying from State with different n");
  copy_threevec (_from.pos, _to.pos);
  copy_threevec (_from.vel, _to.vel);
  copy_threevec (_from.acc, _to.acc);
  _to.jerk = std::vector<T>(_from.jerk.begin(), _from.jerk.end());
}


//
// State for velocity systems like boids/flocking, potential/vortex particle dynamics
//
template <class T, class IT, uint8_t D>
struct VelocityState {

  VectorSoA<T,IT,D> pos, vel;
  std::vector<T> jerk;		// still calling this "jerk", even though it's acceleration magnitude now

  // constructor needs a size
  VelocityState(const IT _n) : pos(_n), vel(_n), jerk(_n) {}

  // alternatively, construct from an existing object
  VelocityState(const VelocityState& _from) :
    pos(_from.pos), vel(_from.vel), jerk(_from.jerk) {}

  // destructor
  ~VelocityState() { }

  void resize(const int _n) {
    pos.resize(_n);
    vel.resize(_n);
    jerk.resize(_n);
  }

  void init_rand(std::mt19937 _gen) {
    pos.init_rand(_gen, -1.0, 1.0);
    vel.zero();
  }

  void zero(const IT _is, const IT _if) {
    vel.zero(_is,_if);
  }

  void zero() {
    zero(0, (IT)pos[0].size());
  }

  void zero_jerk(const IT _is, const IT _if) {
    for (IT i=_is; i<_if; ++i) jerk[i] = 0.0;
  }

  void zero_jerk() {
    zero_jerk(0, (IT)jerk.size());
  }

  void reorder(const std::vector<IT>& _idx) {
    pos.reorder(_idx);
    vel.reorder(_idx);
    std::array<T,_idx.size()> tmp;
    std::copy(jerk.begin(), jerk.end(), tmp.begin());
    for (IT i=0; i<jerk.size(); ++i) jerk[i] = tmp[_idx[i]];
  }

  // computes new position from pos, vel, dt
  void euler_step(const double _dt, const IT _ifirst, const IT _ilast) {
    // first order in velocity
    for (uint8_t d=0; d<D; ++d) {
      for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += _dt*vel.x[d][i];
    }
  }

  // computes new position from pos, vel, dt, using both velocities
  void rk2_step(const double _dt, const IT _ifirst, const IT _ilast, const VelocityState& _alt) {
    // second order in velocity and position
    for (uint8_t d=0; d<D; ++d) {
      for (IT i=_ifirst; i<_ilast; ++i) pos.x[d][i] += 0.5*_dt*(vel.x[d][i]+_alt.vel.x[d][i]);
    }
  }
};


template<class F, class FIT, class T, class TIT, uint8_t D>
void copy_state (const VelocityState<F,FIT,D>& _from, VelocityState<T,TIT,D>& _to) {
  assert(_from.jerk.size() == _to.jerk.size() && "Copying from State with different n");
  copy_threevec (_from.pos, _to.pos);
  copy_threevec (_from.vel, _to.vel);
  _to.jerk = std::vector<T>(_from.jerk.begin(), _from.jerk.end());
}


#ifdef USE_MPI
// non-class method to swap sources
void __attribute__ ((noinline)) send_state(const State& _tosend, const int _n, const int ito, MPI_Request* handle) {
  // do the transfers, saving the handle for future reference
  send_vector(_tosend.pos, _n, ito, handle);
  send_vector(_tosend.vel, _n, ito, handle);
  send_vector(_tosend.acc, _n, ito, handle);
  MPI_Isend(_tosend.jerk, _n, MPI_FLOAT, ito, 10, MPI_COMM_WORLD, &handle);
}

void __attribute__ ((noinline)) receive_state(State& _torecv, const int _n, const int ifrom, MPI_Request* handle) {
  // do the transfers, saving the handle for future reference
  receive_vector(_torecv.pos, _n, ifrom, handle);
  receive_vector(_torecv.vel, _n, ifrom, handle);
  receive_vector(_torecv.acc, _n, ifrom, handle);
  MPI_Irecv(_torecv.jerk, _n, MPI_FLOAT, ifrom, 10, MPI_COMM_WORLD, &handle);
}
#endif
