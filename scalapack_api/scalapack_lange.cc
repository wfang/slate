//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate.hh"
#include "scalapack_slate.hh"
#include <complex>

#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads(int nt);
inline int slate_set_num_blas_threads(const int nt) { return MKL_Set_Num_Threads(nt); }
#else
inline int slate_set_num_blas_threads(const int nt) { return -1; }
#endif

namespace slate {
namespace scalapack_api {

// -----------------------------------------------------------------------------

// Required CBLACS calls
extern "C" void Cblacs_gridinfo(int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);

// Type generic function calls the SLATE routine
template< typename scalar_t >
blas::real_type<scalar_t> slate_plange(const char* normstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)
// Each C interface calls the type generic slate_pher2k

extern "C" blas_float_return PSLANGE(const char* norm, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" blas_float_return pslange(const char* norm, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" blas_float_return pslange_(const char* norm, int* m, int* n, float* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PDLANGE(const char* norm, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlange(const char* norm, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pdlange_(const char* norm, int* m, int* n, double* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" blas_float_return PCLANGE(const char* norm, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" blas_float_return pclange(const char* norm, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" blas_float_return pclange_(const char* norm, int* m, int* n, std::complex<float>* a, int* ia, int* ja, int* desca, float* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------

extern "C" double PZLANGE(const char* norm, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlange(const char* norm, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

extern "C" double pzlange_(const char* norm, int* m, int* n, std::complex<double>* a, int* ia, int* ja, int* desca, double* work)
{
    return slate_plange(norm, *m, *n, a, *ia, *ja, desca, work);
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
blas::real_type<scalar_t> slate_plange(const char* normstr, int m, int n, scalar_t* a, int ia, int ja, int* desca, blas::real_type<scalar_t>* work)
{
    // todo: figure out if the pxq grid is in row or column

    // make blas single threaded
    // todo: does this set the omp num threads correctly
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    lapack::Norm norm = lapack::char2norm(normstr[0]);
    static slate::Target target = slate_scalapack_set_target();
    static int verbose = slate_scalapack_set_verbose();
    int64_t lookahead = 1;

    // Matrix sizes
    int64_t Am = m;
    int64_t An = n;

    // create SLATE matrices from the ScaLAPACK layouts
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(desc_CTXT(desca), &nprow, &npcol, &myrow, &mycol);
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(desc_M(desca), desc_N(desca), a, desc_LLD(desca), desc_MB(desca), nprow, npcol, MPI_COMM_WORLD);
    A = slate_scalapack_submatrix(Am, An, A, ia, ja, desca);

    if (verbose && myrow==0 && mycol==0)
        logprintf("%s\n", "lange");

    blas::real_type<scalar_t> A_norm;
    A_norm = slate::norm(norm, A, {
        {slate::Option::Target, target},
        {slate::Option::Lookahead, lookahead}
    });

    // omp_set_num_threads(saved_num_omp_threads);
    slate_set_num_blas_threads(saved_num_blas_threads);

    return A_norm;
}

} // namespace scalapack_api
} // namespace slate
