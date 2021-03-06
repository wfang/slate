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

///-----------------------------------------------------------------------------
/// \file
///
#ifndef SLATE_CUBLAS_HH
#define SLATE_CUBLAS_HH

#ifdef SLATE_WITH_CUDA
    #include <cublas_v2.h>
#else

#include "slate_cuda.hh"

#include <complex>

typedef void* cublasHandle_t;
typedef int cublasOperation_t;

enum {
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
};

enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS,
    CUBLAS_STATUS_NOT_INITIALIZED,
    CUBLAS_STATUS_ALLOC_FAILED,
    CUBLAS_STATUS_INVALID_VALUE,
    CUBLAS_STATUS_ARCH_MISMATCH,
    CUBLAS_STATUS_MAPPING_ERROR,
    CUBLAS_STATUS_EXECUTION_FAILED,
    CUBLAS_STATUS_INTERNAL_ERROR,
    CUBLAS_STATUS_NOT_SUPPORTED,
    CUBLAS_STATUS_LICENSE_ERROR,
};

#ifdef __cplusplus
extern "C" {
#endif

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* streamId);
cublasStatus_t cublasDasum(
    cublasHandle_t handle, int n, const double* x, int incx, double* result);
cublasStatus_t cublasDestroy(cublasHandle_t handle);

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                               const void *A, int lda, void *B, int ldb);

cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* Aarray[], int lda,
                        const float* Barray[], int ldb,
    const float* beta,        float* Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasDgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha, const double* Aarray[], int lda,
                         const double* Barray[], int ldb,
    const double* beta,        double* Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasCgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex* alpha, const cuComplex* Aarray[], int lda,
                            const cuComplex* Barray[], int ldb,
    const cuComplex* beta,        cuComplex* Carray[], int ldc,
    int batchCount);

cublasStatus_t cublasZgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex* alpha, const cuDoubleComplex* Aarray[], int lda,
                                  const cuDoubleComplex* Barray[], int ldb,
    const cuDoubleComplex* beta,        cuDoubleComplex* Carray[], int ldc,
    int batchCount);

#ifdef __cplusplus
}
#endif

#endif // not SLATE_WITH_CUDA

#endif // SLATE_CUBLAS_HH
