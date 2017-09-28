
CC  = mpicc
CPP = mpicxx

CFLAGS  = -O3 -std=c99
CCFLAGS = -O3 -std=c++17 -fopenmp 

MPI  = /home/pwu11/app/include
CUDA = /usr/local/cuda

INC = -I$(MPI)/include \
      -I$(MKLROOT)/include

LIB = -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl \
      -L$(CUDA)/lib64 \
      -lmpi
