# adapDt
Test program for adaptive time stepping in a gravitational NBody system

## Build and run
With `g++` installed, you can simply run:

	make
	./adapDt -n=10000 -s=5

Generate and save a re-usable set of particles, find the "true" solution, and compare to it later:

	./adapDt -n=10000 -s=0 -o=initial.dat
	./adapDt -n=10000 -s=10 -i=initial.dat -true -o=truesoln.dat
	./adapDt -n=10000 -s=10 -i=initial.dat -c=truesoln.dat

## Description
This is a simple N-Body gravitational direct solver. The aim is to experiment with particle ordering
and methods to adapt the time step by factors of 2 for groups of particles in order to complete
the work faster with similar total error.

# To Do
* Keep the MPI version up-to-date
* Support higher-order forward integrators (Verlet or AB2 come to mind)
* Make the slow-fast splits solution-adaptive (means finding a better estimate of potential error)
* Pull out the classes/structs to new header files
* Consider passing one std::span (c++20) instead of 3 separate values
* Use jerk magnitude as a proxy for error
