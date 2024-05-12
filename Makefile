# GCC compiler suite
CC=g++
# AMD or clang
#CC=clang++

# last is fastest
#CPPOPTS=-O3 -mavx2 -mfma
#CPPOPTS=-Ofast -march=native
CPPOPTS=-O3 -march=native
CPPOPTS+=-fopenmp

HEADERS=VectorSoA.h State.h ParticleSys.h GravKernels.h

#INCLUDE=-I/opt/Vc/include
#LIBS=-L/opt/Vc/lib -lVc

all : adapDt.bin

%.bin : %.cpp VectorSoA.h $(HEADERS)
	$(CC) $(CPPOPTS) $(INCLUDE) -o $@ $< $(LIBS)

clean :
	rm -f multidt1.bin
