CFLAGS  = -std=c99
CCFLAGS = -std=gnu++1y
#--------------------------------------------
# if debug
ifeq (debug, $(filter debug,$(MAKECMDGOALS)))
	CFLAGS += -O0 -g
	CCFLAGS += -O0 -g
else
	CFLAGS += -O2
	CCFLAGS += -O2
endif

#---------------------------------------------
# if OpenMP
ifeq (openmp,$(filter openmp,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_OPENMP
	ifeq (xl,$(filter xl,$(MAKECMDGOALS)))
		CCFLAGS += -qsmp=omp
		CFLAGS += -qsmp=omp
	else
		CCFLAGS += -fopenmp
		CFLAGS += -fopenmp
	endif
else
	SRC += slate_NoOpenmp.cc
endif

#------------------------------------------------------
# if MPI
ifeq (mpi,$(filter mpi,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_MPI
	# LIB += -lmpi
# if Spectrum MPI
else ifeq (spectrum,$(filter spectrum,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_MPI
	LIB += -lmpi_ibm
else
	SRC += slate_NoMpi.cc
endif

#-----------------------------------------------------------------------------
# if MKL 
ifeq (mkl,$(filter mkl,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_MKL
	# if Linux
	ifeq (linux,$(filter linux,$(MAKECMDGOALS)))
		LIB += -L${MKLROOT}/lib \
		       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	# if MacOS
	else ifeq (macos,$(filter macos,$(MAKECMDGOALS)))
		LIB += -L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib \
		       -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
	endif
# if ESSL
else ifeq (essl,$(filter essl,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_ESSL 
	LIB += -lessl -llapacke -llapack
endif

#-----------------------------------------
# if CUDA
ifeq (cuda,$(filter cuda,$(MAKECMDGOALS)))
	CCFLAGS += -DSLATE_WITH_CUDA -I${CUDADIR}/include
	LIB += -L${CUDADIR}/lib64 -lcublas -lcudart
else
	SRC += slate_NoCuda.cc
	SRC += slate_NoCublas.cc
endif

#-------------------------------------------------------------------------------
SRC += blas.cc \
       lapack.cc \
       slate_Matrix_potrf.cc \
       slate_Matrix_syrk.cc \
       slate_Matrix_gemm.cc

OBJ = $(SRC:.cc=.o)
debug:
	@echo "built with debug info (-g -O0)"

openmp:
	@echo built with OpenMP

mpi:
	@echo built with MPI

spectrum:
	@echo built with Spectrum MPI

mkl:
	@echo built with MKL

essl:
	@echo built with ESSL

cuda:
	@echo built with CUDA

xl:
	@echo built with IBM XL

linux macos: $(OBJ) trace/trace.o
	$(CXX) $(CCFLAGS) $(OBJ) app.cc trace/trace.o $(LIB) -o app
	$(CXX) $(CCFLAGS) $(OBJ) xmm.cc trace/trace.o $(LIB) -o xmm

clean:
	rm -f $(OBJ)
	rm -f app app.o trace_*.svg

trace/trace.o: trace/trace.c
	$(CC) $(CFLAGS) -DMPI -c trace/trace.c -o trace/trace.o

.cc.o:
	$(CXX) $(CCFLAGS) -c $< -o $@
