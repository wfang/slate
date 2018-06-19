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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to htts://grous.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#include "slate.hh"
#include "slate_cuda.hh"
#include "blas_fortran.hh"
#include <complex>

#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads(int nt);
inline int slate_set_num_blas_threads(const int nt) { return MKL_Set_Num_Threads(nt); }
#else
inline int slate_set_num_blas_threads(const int nt) { return 1; }
#endif

namespace slate {
namespace lapack_api {

// -----------------------------------------------------------------------------

void cblas_ssyr2k(const CBLAS_LAYOUT Layout, const CBLAS_UPLO uplo, const CBLAS_TRANSPOSE trans, const MKL_INT n, const MKL_INT k, const float alpha, const float* a, const MKL_INT lda, const float* b, const MKL_INT ldb, const float beta, float* c, const MKL_INT ldc);

// Local function
template< typename scalar_t >
void slate_syrk(const char* uplostr, const char* transastr, const int n, const int k, const scalar_t alpha, scalar_t* a, const int lda, const scalar_t beta, scalar_t* c, const int ldc);

// -----------------------------------------------------------------------------
// C interfaces (FORTRAN_UPPER, FORTRAN_LOWER, FORTRAN_UNDERSCORE)

#define slate_ssyrk BLAS_FORTRAN_NAME( slate_ssyrk, SLATE_SSYRK )
#define slate_dsyrk BLAS_FORTRAN_NAME( slate_dsyrk, SLATE_DSYRK )
#define slate_csyrk BLAS_FORTRAN_NAME( slate_csyrk, SLATE_CSYRK )
#define slate_zsyrk BLAS_FORTRAN_NAME( slate_zsyrk, SLATE_ZSYRK )

extern "C" void slate_ssyrk(const char* uplo, const char* transa, const int* n, const int* k, const float* alpha, float* a, const int* lda, const float* beta, float* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

extern "C" void slate_dsyrk(const char* uplo, const char* transa, const int* n, const int* k, const double* alpha, double* a, const int* lda, const double* beta, double* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

extern "C" void slate_csyrk(const char* uplo, const char* transa, const int* n, const int* k, const std::complex<float>* alpha, std::complex<float>* a, const int* lda, const std::complex<float>* beta, std::complex<float>* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

extern "C" void slate_zsyrk(const char* uplo, const char* transa, const int* n, const int* k, const std::complex<double>* alpha, std::complex<double>* a, const int* lda, const std::complex<double>* beta, std::complex<double>* c, const int* ldc)
{
    slate_syrk(uplo, transa, *n, *k, *alpha, a, *lda, *beta, c, *ldc);
}

// -----------------------------------------------------------------------------

// Type generic function calls the SLATE routine
template< typename scalar_t >
void slate_syrk(const char* uplostr, const char* transastr, const int n, const int k, const scalar_t alpha, scalar_t* a, const int lda, const scalar_t beta, scalar_t* c, const int ldc)
{
    // Check and initialize MPI, else SLATE calls to MPI will fail
    int initialized, provided;
    assert(MPI_Initialized(&initialized) == MPI_SUCCESS);
    if (! initialized) assert(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS);

    // todo: does this set the omp num threads correctly in all circumstances
    int saved_num_blas_threads = slate_set_num_blas_threads(1);

    blas::Uplo uplo = blas::char2uplo(uplostr[0]);
    blas::Op transA = blas::char2op(transastr[0]);
    int64_t lookahead = 1;
    int64_t p = 1;
    int64_t q = 1;
    static slate::Target target;
    static int64_t target_is_set = 0;
    static int64_t nb = 0;

    // set target if not already done
    if (! target_is_set) {
        int cudadevcount = 0;
        target = slate::Target::HostTask;
        if (std::getenv("SLATE_TARGET")) {
            char targetchar = (char)(toupper(std::getenv("SLATE_TARGET")[4]));
            if (targetchar=='T') target = slate::Target::HostTask;
            else if (targetchar=='N') target = slate::Target::HostNest;
            else if (targetchar=='B') target = slate::Target::HostBatch;
            else if (targetchar=='C') target = slate::Target::Devices;
        }
        else if (cudaGetDeviceCount(&cudadevcount)==cudaSuccess && cudadevcount>0)
            target = slate::Target::Devices;
        target_is_set = 1;
    }

    // set nb if not already done
    if (nb == 0) {
        nb = 256;
        if (std::getenv("SLATE_NB"))
            nb = strtol(std::getenv("SLATE_NB"), NULL, 0);
        else if (target==slate::Target::HostTask)
            nb = 512;
        else if (target==slate::Target::Devices)
            nb = 1024;
    }

    // setup so op(A) is n-by-k
    int64_t Am = (transA == blas::Op::NoTrans ? n : k);
    int64_t An = (transA == blas::Op::NoTrans ? k : n);
    int64_t Cn = n;

    // create SLATE matrices from the LAPACK data
    auto A = slate::Matrix<scalar_t>::fromLAPACK(Am, An, a, lda, nb, p, q, MPI_COMM_WORLD);
    auto C = slate::SymmetricMatrix<scalar_t>::fromLAPACK(uplo, Cn, c, ldc, nb, p, q, MPI_COMM_WORLD);

    if (transA == blas::Op::Trans)
        A = transpose(A);
    else if (transA == blas::Op::ConjTrans)
        A = conj_transpose(A);
    assert(A.mt() == C.mt());

    // typedef long long lld;
    // printf("SLATE SYRK n %lld k %lld nb %lld \n", (lld)n, (lld)k, (lld)nb);
    slate::syrk(alpha, A, beta, C, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target}
    });

    slate_set_num_blas_threads(saved_num_blas_threads);
}

} // namespace lapack_api
} // namespace slate