SLATE SCALAPACK COMPATIBILITY API
=================================

This API is designed to provide a high level of compatibility for
ScaLAPACK users who would like to take advantage of SLATE.  Calls to
ScaLAPACK routines are captured by the API and a lightweight SLATE
matrix object is constructed to point to the pre-existing ScaLAPACK
data.  If desired, SLATE can transparently use available GPUs to
execute the available routines.  Any calls that are missing in this
API should fall through to the ScaLAPACK implementation.

NOTE: The ScaLAPACK BLACS grid needs to be Column-major.
CALL BLACS_GRIDINIT( ICTXT, 'Col-major', NPROW, NPCOL )

NOTE: The ScaLAPACK blocking (NB_,MB_) is used as the SLATE block size
nb.  However, SLATE may require larger block sizes than ScaLAPACK to
get performance.  Please check that the ScaLAPACK matrices have the
appropriate blocking to enable good SLATE performance.

NOTE: A submatrix must start and end on a tile boundary.
Taking a submatrix of global matrix A of size Am,An starting at ia,ij.

slate_scalapack_submatrix(int Am, int An, slate::Matrix<scalar_t>& A, int ia, int ja, int* desca)

The ia and ja must be devisable by A[MB_] and A[NB_] respectively.
    assert((ia-1) % desca[MB_]==0);
    assert((ja-1) % desca[NB_]==0);
And any submatrix must end on a tile boundary.
    assert(Am % desca[MB_]==0);
    assert(An % desca[NB_]==0);


USING THE COMPATIBILITY API
---------------------------

* RUNTIME VIA PRELOAD: A user can preload this library for runtime
interception.

env LD_PRELOAD=$SLATE_DIR/lib/libslate_scalapack_api.so mpirun -np 4 $SLATE_DIR/test/test gemm --p 2 --q 2

* COMPILE TIME VIA LINKING: A user can link with this library to get
link time binding.  Remember that this library depends on (precedes)
SLATE and needs to handle some of the (precede) ScaLAPACK calls.  For
example:

${LINK} *.o -lslate_scalapack_api -lslate -lmkl_scalapack_lp64 ... -lpthread -lm -ldl -lcublas -lcudart -o ${EXE}


ENVIRONMENT VARIABLES
---------------------

The library libslate_scalapack_api.so uses some environment variables
to make decisions that are not available in a the ScaLAPACK
parameters.

* SLATE_SCALAPACK_TARGET  HostTask (default), Devices, HostNest, HostBatch (case indifferent)
* SLATE_SCALAPACK_VERBOSE  0,1 (0: no output,  1: print some minor output)

Example on a properly configured SLATE install on a machine with GPUs.

Run the reference test using ScaLAPACK

${SLATE_DIR}/test/test gemm --p 1 --q 1 --ref y

Run the reference test, calling ScaLAPACK, which gets intercepted and sent to SLATE/Devices

env SLATE_SCALAPACK_TARGET=Devices SLATE_VERBOSE=1 ${SLATE_DIR}/test/test gemm --p 1 --q 1 --ref y


TESTING
-------

The SLATE tester can be used to run tests.  The verbose flag will
print a small message, otherwise it is not easy to tell if the routine
was intercepted.  Using the SLATE tester will not work for norms,
because there are temporary implementations of norms within SLATE.

env LD_PRELOAD=$SLATE_DIR/lib/libslate_scalapack_api.so SLATE_VERBOSE=1 SLATE_SCALAPACK_TARGET=HostTask $SLATE_DIR/test/test gemm

env LD_PRELOAD=$SLATE_DIR/lib/libslate_scalapack_api.so SLATE_VERBOSE=1 SLATE_SCALAPACK_TARGET=Devices $SLATE_DIR/test/test gemm

Testing can also be done with the ScaLAPACK and PBLAS testers.  Note,
SLATE does not handle parameter errors, so error exits will need to be
disabled in the test input.

PxBLAS3TST.dat line 6 set to:     F  logical flag, T to test error exits

Other test input file changes may be needed to make sure that the
matrix/tile boundaries meets SLATE's requirements.

cd ${PBLAS_DIR}/TESTING/; env LD_PRELOAD=${SLATE_DIR}/lib/libslate_scalapack_api.so mpirun -np 4 -envall ./xspblas3tst


