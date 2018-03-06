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
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_blas.hh"
#include "slate_internal.hh"
#include "slate_internal_batch.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void herk(typename blas::traits<scalar_t>::real_t alpha, Matrix< scalar_t >&& A,
          typename blas::traits<scalar_t>::real_t beta,  HermitianMatrix< scalar_t >&& C,
          int priority)
{
    herk(internal::TargetType<target>(),
         alpha, A,
         beta,  C,
         priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Host OpenMP task implementation.
template <typename scalar_t>
void herk(internal::TargetType<Target::HostTask>,
          typename blas::traits<scalar_t>::real_t alpha, Matrix< scalar_t >& A,
          typename blas::traits<scalar_t>::real_t beta,  HermitianMatrix< scalar_t >& C,
          int priority)
{
    using real_t = typename blas::traits<scalar_t>::real_t;

    scalar_t alpha_ = scalar_t(alpha);
    scalar_t beta_  = scalar_t(beta);

    // Lower, NoTrans
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = j; i < C.mt(); ++i)  // lower
            if (C.tileIsLocal(i, j)) {
                if (i == j) {
                    #pragma omp task shared(A, C) priority(priority)
                    {
                        A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                        C.tileMoveToHost(j, j, C.tileDevice(j, j));
                        herk(real_t(-1.0), A(j, 0),
                             beta,         C(j, j));
                        A.tileTick(j, 0);
                        A.tileTick(j, 0);
                    }
                }
                else {
                    #pragma omp task shared(A, C) priority(priority)
                    {
                        A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                        A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                        C.tileMoveToHost(i, j, C.tileDevice(i, j));
                        auto Aj0 = A(j, 0);
                        gemm(alpha_, A(i, 0),
                                     conj_transpose(Aj0),
                             beta_,  C(i, j));
                        A.tileTick(i, 0);
                        A.tileTick(j, 0);
                    }
                }
            }

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Host nested OpenMP task implementation.
template <typename scalar_t>
void herk(internal::TargetType<Target::HostNest>,
          typename blas::traits<scalar_t>::real_t alpha, Matrix< scalar_t >& A,
          typename blas::traits<scalar_t>::real_t beta,  HermitianMatrix< scalar_t >& C,
          int priority)
{
    using real_t = typename blas::traits<scalar_t>::real_t;

    scalar_t alpha_ = scalar_t(alpha);
    scalar_t beta_  = scalar_t(beta);

    // Lower, NoTrans
    for (int64_t j = 0; j < C.nt(); ++j)
        if (C.tileIsLocal(j, j))
            #pragma omp task shared(A, C)
            {
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(j, j, C.tileDevice(j, j));
                herk(real_t(-1.0), A(j, 0),
                     beta,         C(j, j));
                A.tileTick(j, 0);
                A.tileTick(j, 0);
            }

//  #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t j = 0; j < C.nt(); ++j)
        for (int64_t i = 0; i < C.mt(); ++i)  // full
            if (i >= j+1)                     // lower
                if (C.tileIsLocal(i, j))
                {
                    A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                    A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                    C.tileMoveToHost(i, j, C.tileDevice(i, j));
                    auto Aj0 = A(j, 0);
                    gemm(alpha_, A(i, 0),
                                 conj_transpose(Aj0),
                         beta_,  C(i, j));
                    A.tileTick(i, 0);
                    A.tileTick(j, 0);
                }

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// Host batched implementation.
template <typename scalar_t>
void herk(internal::TargetType<Target::HostBatch>,
          typename blas::traits<scalar_t>::real_t alpha, Matrix< scalar_t >& A,
          typename blas::traits<scalar_t>::real_t beta,  HermitianMatrix< scalar_t >& C,
          int priority)
{
    using real_t = typename blas::traits<scalar_t>::real_t;

    // diagonal tiles by herk on host
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C)
            {
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(j, j, C.tileDevice(j, j));
                herk(real_t(-1.0), A(j, 0),
                     beta,         C(j, j));
                A.tileTick(j, 0);
                A.tileTick(j, 0);
            }
        }
    }

    // load off-diagonal tiles to host, if not there
    // also count tiles
    int batch_count = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j+1; i < C.mt(); ++i) {  // lower
            if (C.tileIsLocal(i, j)) {
                A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(i, j, C.tileDevice(i, j));
                ++batch_count;
            }
        }
    }
    if (batch_count > 0) {
        // off-diagonal tiles by batch gemm on host
        CBLAS_TRANSPOSE opA = (A.op() == Op::NoTrans ? CblasNoTrans : CblasConjTrans);
        CBLAS_TRANSPOSE opB = (A.op() == Op::NoTrans ? CblasConjTrans : CblasNoTrans);
        std::vector< CBLAS_TRANSPOSE > opA_array( batch_count, opA );  // all same
        std::vector< CBLAS_TRANSPOSE > opB_array( batch_count, opB );  // all same
        std::vector< int > m_array( batch_count );
        std::vector< int > n_array( batch_count );
        std::vector< int > k_array( batch_count );
        std::vector< scalar_t > alpha_array( batch_count, alpha );  // all same
        std::vector< scalar_t >  beta_array( batch_count,  beta );  // all same
        std::vector< const scalar_t* > a_array( batch_count );
        std::vector< const scalar_t* > b_array( batch_count );
        std::vector< scalar_t* > c_array( batch_count );
        std::vector< int > lda_array( batch_count );
        std::vector< int > ldb_array( batch_count );
        std::vector< int > ldc_array( batch_count );
        std::vector< int > group_size( batch_count, 1 );  // all same

        int index = 0;
        for (int64_t j = 0; j < C.nt(); ++j) {
            for (int64_t i = j+1; i < C.mt(); ++i) {  // lower
                if (C.tileIsLocal(i, j)) {
                    m_array[ index ] = C(i, j).mb();
                    n_array[ index ] = C(i, j).nb();
                    k_array[ index ] = A(i, 0).nb();  // should be all same

                    assert( A(i, 0).mb() == m_array[ index ] );
                    assert( A(j, 0).mb() == n_array[ index ] );
                    assert( A(j, 0).nb() == k_array[ index ] );

                    a_array[ index ] = A(i, 0).data();
                    b_array[ index ] = A(j, 0).data();
                    c_array[ index ] = C(i, j).data();

                    lda_array[ index ] = A(i, 0).stride();
                    ldb_array[ index ] = A(j, 0).stride();
                    ldc_array[ index ] = C(i, j).stride();

                    ++index;
                }
            }
        }

        {
            trace::Block trace_block("cblas_dgemm_batch");
            #ifdef SLATE_WITH_MKL
                // mkl_set_num_threads_local(...);
                cblas_gemm_batch( CblasColMajor, opA_array.data(), opB_array.data(),
                                   m_array.data(), n_array.data(), k_array.data(),
                                   alpha_array.data(),
                                   a_array.data(), lda_array.data(),
                                   b_array.data(), ldb_array.data(),
                                   beta_array.data(),
                                   c_array.data(), ldc_array.data(),
                                   batch_count, group_size.data() );
                // mkl_set_num_threads_local(1);
            #else
                assert(false);
            #endif
        }

        for (int64_t j = 0; j < C.nt(); ++j) {
            for (int64_t i = j+1; i < C.mt(); ++i) {  // lower
                if (C.tileIsLocal(i, j)) {
                    A.tileTick(i, 0);
                    A.tileTick(j, 0);
                }
            }
        }
    }

    #pragma omp taskwait
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-k update of single block column (i.e., k = nb).
/// GPU device batched cuBLAS implementation.
template <typename scalar_t>
void herk(internal::TargetType<Target::Devices>,
          typename blas::traits<scalar_t>::real_t alpha, Matrix< scalar_t >& A,
          typename blas::traits<scalar_t>::real_t beta,  HermitianMatrix< scalar_t >& C,
          int priority)
{
    using real_t = typename blas::traits<scalar_t>::real_t;

    // Lower, NoTrans
    assert(C.uplo() == Uplo::Lower);
    assert(A.op() == Op::NoTrans);

    // off-diagonal tiles by batch gemm on device
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, C) priority(1)
        {
            scalar_t** a_array_host = C.a_array_host(device);
            scalar_t** b_array_host = C.b_array_host(device);
            scalar_t** c_array_host = C.c_array_host(device);

            int64_t batch_count = 0;
            for (int64_t j = 0; j < C.nt(); ++j) {
                for (int64_t i = j+1; i < C.mt(); ++i) {  // lower
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            A.tileCopyToDevice(i, 0, device);
                            A.tileCopyToDevice(j, 0, device);
                            C.tileMoveToDevice(i, j, device);
                            a_array_host[ batch_count ] = A(i, 0, device).data();
                            b_array_host[ batch_count ] = A(j, 0, device).data();
                            c_array_host[ batch_count ] = C(i, j, device).data();
                            ++batch_count;
                        }
                    }
                }
            }

            scalar_t** a_array_dev = C.a_array_device(device);
            scalar_t** b_array_dev = C.b_array_device(device);
            scalar_t** c_array_dev = C.c_array_device(device);
            cudaError_t error;
            error = cudaSetDevice(device);
            assert(error == cudaSuccess);

            // cublas_handle uses this stream
            cudaStream_t stream = C.compute_stream(device);
            cublasHandle_t cublas_handle = C.cublas_handle(device);

            error = cudaMemcpyAsync(a_array_dev, a_array_host,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(b_array_dev, b_array_host,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream);
            assert(error == cudaSuccess);

            error = cudaMemcpyAsync(c_array_dev, c_array_host,
                                    sizeof(scalar_t*)*batch_count,
                                    cudaMemcpyHostToDevice,
                                    stream);
            assert(error == cudaSuccess);

            {
                scalar_t alpha_ = scalar_t(alpha);
                scalar_t beta_  = scalar_t(beta);

                trace::Block trace_block("cublasDgemmBatched");
                int nb = C.tileNb(0);
                cublasOperation_t opa = (A.op() == Op::NoTrans ? CUBLAS_OP_N : CUBLAS_OP_C);
                cublasOperation_t opb = (A.op() == Op::NoTrans ? CUBLAS_OP_C : CUBLAS_OP_N);
                cublasStatus_t status =
                    cublasGemmBatched(
                        cublas_handle,  // uses stream
                        opa, opb,
                        nb, nb, nb,
                        &alpha_, (const scalar_t**) a_array_dev, nb,
                                 (const scalar_t**) b_array_dev, nb,
                        &beta_,  c_array_dev, nb,
                        batch_count);
                assert(status == CUBLAS_STATUS_SUCCESS);
                error = cudaStreamSynchronize(stream);
                assert(error == cudaSuccess);
            }

            for (int64_t j = 0; j < C.nt(); ++j) {
                for (int64_t i = j+1; i < C.mt(); ++i) {  // lower
                    if (C.tileIsLocal(i, j)) {
                        if (device == C.tileDevice(i, j)) {
                            //A.tileErase(i, 0, device);  // todo: why? shouldn't tileTick deal with this?
                            //A.tileErase(j, 0, device);  // ditto
                            A.tileTick(i, 0);
                            A.tileTick(j, 0);
                        }
                    }
                }
            }
        }
    }

    // diagonal tiles by herk on host
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C)
            {
                A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                C.tileMoveToHost(j, j, C.tileDevice(j, j));
                herk(real_t(-1.0), A(j, 0),
                     beta,         C(j, j));
                A.tileTick(j, 0);
                A.tileTick(j, 0);
            }
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void herk< Target::HostTask, float >(
    float alpha, Matrix< float >&& A,
    float beta,  HermitianMatrix< float >&& C,
    int priority);

template
void herk< Target::HostNest, float >(
    float alpha, Matrix< float >&& A,
    float beta,  HermitianMatrix< float >&& C,
    int priority);

template
void herk< Target::HostBatch, float >(
    float alpha, Matrix< float >&& A,
    float beta,  HermitianMatrix< float >&& C,
    int priority);

template
void herk< Target::Devices, float >(
    float alpha, Matrix< float >&& A,
    float beta,  HermitianMatrix< float >&& C,
    int priority);

// ----------------------------------------
template
void herk< Target::HostTask, double >(
    double alpha, Matrix< double >&& A,
    double beta,  HermitianMatrix< double >&& C,
    int priority);

template
void herk< Target::HostNest, double >(
    double alpha, Matrix< double >&& A,
    double beta,  HermitianMatrix< double >&& C,
    int priority);

template
void herk< Target::HostBatch, double >(
    double alpha, Matrix< double >&& A,
    double beta,  HermitianMatrix< double >&& C,
    int priority);

template
void herk< Target::Devices, double >(
    double alpha, Matrix< double >&& A,
    double beta,  HermitianMatrix< double >&& C,
    int priority);

// ----------------------------------------
template
void herk< Target::HostTask, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority);

template
void herk< Target::HostNest, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority);

template
void herk< Target::HostBatch, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority);

template
void herk< Target::Devices, std::complex<float> >(
    float alpha, Matrix< std::complex<float> >&& A,
    float beta,  HermitianMatrix< std::complex<float> >&& C,
    int priority);

// ----------------------------------------
template
void herk< Target::HostTask, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority);

template
void herk< Target::HostNest, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority);

template
void herk< Target::HostBatch, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority);

template
void herk< Target::Devices, std::complex<double> >(
    double alpha, Matrix< std::complex<double> >&& A,
    double beta,  HermitianMatrix< std::complex<double> >&& C,
    int priority);

} // namespace internal
} // namespace slate