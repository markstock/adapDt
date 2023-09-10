# adapDt
Test program for adaptive time stepping in a gravitational NBody system

## Build and run
With `g++` installed, you can simply run

	make
	./adapDt -n=10000 -s=5

## Description
This is a simple N-Body gravitational direct solver. The aim is to experiment with particle ordering
and methods to adapt the time step by factors of 2 for groups of particles in order to complete
the work faster with similar total error.

