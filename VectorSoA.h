/*
 * VectorSoA.h - header for arrays of vectors
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


//
// Class to store a D-vec for each particle in SoA style
// T is float storage class
// IT is integer (index) storage class (typically uint32_t or size_t)
// D is number of spatial dimensions
//
template <class T, class IT, uint8_t D>
struct VectorSoA {

  // array of D vectors of type T
  std::array<std::vector<T>,D> x;

  // constructor needs a size
  VectorSoA(const IT _n) {
    for (uint8_t d=0; d<D; ++d) x[d].resize(_n);
    zero();
  }

  // alternatively, construct from an existing object
  VectorSoA(const VectorSoA& _from) : x(_from.x) {}

  // destructor
  ~VectorSoA() { }

  void init_rand(std::mt19937 _gen, const T _min, const T _max) {
    std::uniform_real_distribution<> zmean_dist(_min, _max);
    const IT n = x[0].size();
    for (uint8_t d=0; d<D; ++d) for (IT i=0; i<n; ++i) x[d][i] = zmean_dist(_gen);
  }

  void zero(const IT _is, const IT _if) {
    for (uint8_t d=0; d<D; ++d) for (IT i=_is; i<_if; ++i) x[d][i] = 0.0;
  }

  void zero() {
    zero((IT)0, x[0].size());
  }

  void set_from(const IT _is, const IT _if, const VectorSoA<T,IT,D>& _val, const IT _vs, const IT _vf) {
    //printf("copying from %d - %d (of %ld) into %d - %d (of %ld)\n", _vs, _vf, _val.x.size(), _is, _if, x.size());
    for (uint8_t d=0; d<D; ++d) for (IT i=_is; i<_if; ++i) x[d][i] = _val.x[d][_vs+i-_is];
  }

  void add_from(const IT _is, const IT _if, const VectorSoA<T,IT,D>& _val) {
    for (uint8_t d=0; d<D; ++d) for (IT i=_is; i<_if; ++i) x[d][i] += _val.x[d][i];
  }

  void copy_to(const IT _is, const IT _if, VectorSoA<T,IT,D>& _val) {
    for (uint8_t d=0; d<D; ++d) for (IT i=_is; i<_if; ++i) _val.x[d][i] += x[d][i];
  }

  void reorder(const std::vector<IT>& _idx) {
    const IT n = x[0].size();
    std::vector<T> tmp(n);
    for (uint8_t d=0; d<D; ++d) {
      assert(_idx.size() == x[d].size() && "Idx and X vectors do not match in size");
      std::copy(x[d].begin(), x[d].end(), tmp.begin());
      for (IT i=0; i<n; ++i) x[d][i] = tmp[_idx[i]];
    }
  }
};


//
// Copy one D-Vec to another with different template param
//
template<class F, class FIT, class T, class TIT, uint8_t D>
void copy_threevec (const VectorSoA<F,FIT,D>& _from, VectorSoA<T,TIT,D>& _to) {
  assert(_from.x.size() == _to.x.size() && "Copying from VectorSoA with different D");
  assert(_from.x[0].size() == _to.x[0].size() && "Copying from VectorSoA with different n");
  for (uint8_t d=0; d<D; ++d) {
    _to.x[d] = std::vector<T>(_from.x[d].begin(), _from.x[d].end());
  }
}


#ifdef USE_MPI
// non-class method to swap sources
void __attribute__ ((noinline)) send_vector(const VectorSoA& _tosend, const int _n, const int ito, MPI_Request* handle) {
  for (uint8_t d=0; d<D; ++d) {
    // do the transfers, saving the handle for future reference
    MPI_Isend(_tosend.x[d], _n, MPI_FLOAT, ito, 10+d, MPI_COMM_WORLD, &handle[d]);
  }
}

void __attribute__ ((noinline)) receive_vector(VectorSoA& _torecv, const int _n, const int ifrom, MPI_Request* handle) {
  for (uint8_t d=0; d<D; ++d) {
    // do the transfers, saving the handle for future reference
    MPI_Irecv(_torecv.x[d], _n, MPI_FLOAT, ifrom, 10+d, MPI_COMM_WORLD, &handle[d]);
  }
}
#endif

