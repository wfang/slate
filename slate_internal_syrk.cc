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
#include "slate_SymmetricMatrix.hh"
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
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Dispatches to target implementations.
/// C is Lower, NoTrans or Upper, Trans/ConjTrans.
/// In complex case, A and C cannot be ConjTrans.
template <Target target, typename scalar_t>
void syrk(scalar_t alpha, Matrix<scalar_t>&& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>&& C,
          int priority)
{
    if (! ( ( C.uplo_logical() == Uplo::Lower )
            &&
            ( C.is_real || (C.op() != Op::ConjTrans &&
                            A.op() != Op::ConjTrans) ) ) )
    {
        throw std::exception();
    }

    syrk(internal::TargetType<target>(),
         alpha, A,
         beta,  C,
         priority);
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host OpenMP task implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostTask>,
          scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    // Lower, NoTrans
    int err = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j; i < C.mt(); ++i) {  // lower
            if (C.tileIsLocal(i, j)) {
                if (i == j) {
                    #pragma omp task shared(A, C, err) priority(priority)
                    {
                        try {
                            A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                            C.tileMoveToHost(j, j, C.tileDevice(j, j));
                            syrk(alpha, A(j, 0),
                                 beta,  C(j, j));
                            A.tileTick(j, 0);
                            A.tileTick(j, 0);
                        }
                        catch (std::exception& e) {
                            err = __LINE__;
                        }
                    }
                }
                else {
                    #pragma omp task shared(A, C, err) priority(priority)
                    {
                        try {
                            A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                            A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                            C.tileMoveToHost(i, j, C.tileDevice(i, j));
                            auto Aj0 = A(j, 0);
                            gemm(alpha, A(i, 0),
                                        transpose(Aj0),
                                 beta,  C(i, j));
                            A.tileTick(i, 0);
                            A.tileTick(j, 0);
                        }
                        catch (std::exception& e ) {
                            err = __LINE__;
                        }
                    }
                }
            }
        }
    }

    #pragma omp taskwait

    if (err) {
        throw std::exception();
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host nested OpenMP implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostNest>,
          scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    // Lower, NoTrans
    int err = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C, err)
            {
                try {
                    A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                    C.tileMoveToHost(j, j, C.tileDevice(j, j));
                    syrk(alpha, A(j, 0),
                         beta,  C(j, j));
                    A.tileTick(j, 0);
                    A.tileTick(j, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }

//  #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(...)
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = 0; i < C.mt(); ++i) {  // full
            if (i >= j+1) {                     // strictly lower
                if (C.tileIsLocal(i, j)) {
                    try {
                        A.tileCopyToHost(i, 0, A.tileDevice(i, 0));
                        A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                        C.tileMoveToHost(i, j, C.tileDevice(i, j));
                        auto Aj0 = A(j, 0);
                        gemm(alpha, A(i, 0),
                                    transpose(Aj0),
                             beta,  C(i, j));
                        A.tileTick(i, 0);
                        A.tileTick(j, 0);
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }

    #pragma omp taskwait

    if (err) {
        throw std::exception();
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// Host batched implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
template <typename scalar_t>
void syrk(internal::TargetType<Target::HostBatch>,
          scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    // diagonal tiles by syrk on host
    int err = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C, err)
            {
                try {
                    A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                    C.tileMoveToHost(j, j, C.tileDevice(j, j));
                    syrk(alpha, A(j, 0),
                         beta,  C(j, j));
                    A.tileTick(j, 0);
                    A.tileTick(j, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }

    // load off-diagonal tiles to host, if not there
    // also count tiles
    int batch_count = 0;
    for (int64_t j = 0; j < C.nt(); ++j) {
        for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
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
        Op opA = A.op();
        if (C.op() != Op::NoTrans) {
            if (A.op() == Op::NoTrans)
                opA = C.op();
            else if (A.op() == C.op() || C.is_real) {
                // A and C are both Trans or both ConjTrans; Trans == ConjTrans if real
                opA = Op::NoTrans;
            }
            else
                throw std::exception();
        }

        Op opB = (opA == Op::NoTrans ? Op::Trans : Op::NoTrans);

        std::vector<CBLAS_TRANSPOSE> opA_array(batch_count, cblas_trans_const(opA));  // all same
        std::vector<CBLAS_TRANSPOSE> opB_array(batch_count, cblas_trans_const(opB));  // all same
        std::vector<int> m_array(batch_count);
        std::vector<int> n_array(batch_count);
        std::vector<int> k_array(batch_count);
        std::vector<scalar_t> alpha_array(batch_count, alpha);  // all same
        std::vector<scalar_t>  beta_array(batch_count,  beta);  // all same
        std::vector<const scalar_t*> a_array(batch_count);
        std::vector<const scalar_t*> b_array(batch_count);
        std::vector<scalar_t*> c_array(batch_count);
        std::vector<int> lda_array(batch_count);
        std::vector<int> ldb_array(batch_count);
        std::vector<int> ldc_array(batch_count);
        std::vector<int> group_size(batch_count, 1);  // all same

        int index = 0;
        for (int64_t j = 0; j < C.nt(); ++j) {
            for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
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

        if (C.op() != Op::NoTrans) {
            // swap A <=> B; swap m <=> n
            swap(opA_array, opB_array);
            swap(a_array,   b_array  );
            swap(lda_array, ldb_array);
            swap(m_array,   n_array  );
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
            for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
                if (C.tileIsLocal(i, j)) {
                    A.tileTick(i, 0);
                    A.tileTick(j, 0);
                }
            }
        }
    }

    #pragma omp taskwait

    if (err) {
        throw std::exception();
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update of single block column (i.e., k = nb).
/// GPU device batched cuBLAS implementation.
/// Assumes A is NoTrans or Trans; C is Lower, NoTrans or Upper, Trans.
template <typename scalar_t>
void syrk(internal::TargetType<Target::Devices>,
          scalar_t alpha, Matrix<scalar_t>& A,
          scalar_t beta,  SymmetricMatrix<scalar_t>& C,
          int priority)
{
    int err = 0;
    using std::swap;

    assert(C.num_devices() > 0);

    // off-diagonal tiles by batch gemm on device
    for (int device = 0; device < C.num_devices(); ++device) {
        #pragma omp task shared(A, C, err) priority(priority)
        {
            try {
                // if op(C) is NoTrans, invert opA, opB if possible
                Op opA = A.op();
                if (C.op() != Op::NoTrans) {
                    if (A.op() == Op::NoTrans)
                        opA = C.op();
                    else if (A.op() == C.op() || C.is_real) {
                        // A and C are both Trans or both ConjTrans; Trans == ConjTrans if real
                        opA = Op::NoTrans;
                    }
                    else
                        throw std::exception();
                }

                Op opB = (opA == Op::NoTrans ? Op::Trans : Op::NoTrans);

                for (int64_t j = 0; j < C.nt()-1; ++j) {
                    for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
                        if (C.tileIsLocal(i, j)) {
                            if (device == C.tileDevice(i, j)) {
                                A.tileCopyToDevice(i, 0, device);
                                A.tileCopyToDevice(j, 0, device);
                                C.tileMoveToDevice(i, j, device);
                            }
                        }
                    }
                }

                scalar_t** a_array_host = C.a_array_host(device);
                scalar_t** b_array_host = C.b_array_host(device);
                scalar_t** c_array_host = C.c_array_host(device);

                int64_t batch_count = 0;
                int64_t batch_count_00 = 0;
                int64_t lda00 = 0;
                int64_t ldb00 = 0;
                int64_t ldc00 = 0;
                int64_t mb00 = C.tileMb(0);
                int64_t nb00 = C.tileNb(0);
                int64_t kb = A.tileNb(0);   // == A.tileMb(0)
                for (int64_t j = 0; j < C.nt()-1; ++j) {
                    for (int64_t i = j+1; i < C.mt()-1; ++i) {  // strictly lower
                        if (C.tileIsLocal(i, j)) {
                            if (device == C.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, 0, device).data();
                                b_array_host[batch_count] = A(j, 0, device).data();
                                c_array_host[batch_count] = C(i, j, device).data();
                                lda00 = A(i, 0, device).stride();
                                ldb00 = A(j, 0, device).stride();
                                ldc00 = C(i, j, device).stride();
                                ++batch_count_00;
                                ++batch_count;
                            }
                        }
                    }
                }

                int64_t batch_count_10 = 0;
                int64_t lda10 = 0;
                int64_t ldb10 = 0;
                int64_t ldc10 = 0;
                int64_t mb10 = C.tileMb(C.mt()-1);
                int64_t nb10 = C.tileNb(0);
                // same kb as above
                {
                    int64_t i = C.mt()-1;
                    for (int64_t j = 0; j < C.nt()-1; ++j) {
                        if (C.tileIsLocal(i, j)) {
                            if (device == C.tileDevice(i, j)) {
                                a_array_host[batch_count] = A(i, 0, device).data();
                                b_array_host[batch_count] = A(j, 0, device).data();
                                c_array_host[batch_count] = C(i, j, device).data();
                                lda10 = A(i, 0, device).stride();
                                ldb10 = A(j, 0, device).stride();
                                ldc10 = C(i, j, device).stride();
                                ++batch_count_10;
                                ++batch_count;
                            }
                        }
                    }
                }

                if (C.op() != Op::NoTrans) {
                    // swap A <=> B; swap m <=> n
                    swap(opA, opB);
                    swap(a_array_host, b_array_host);
                    swap(lda00, ldb00);
                    swap(lda10, ldb10);
                    swap(mb00, nb00);
                    swap(mb10, nb10);
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
                    trace::Block trace_block("cublasDgemmBatched");
                    if (batch_count_00 > 0) {
                        cublasStatus_t status =
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opA), cublas_op_const(opB),
                                mb00, nb00, kb,
                                &alpha, (const scalar_t**) a_array_dev, lda00,
                                        (const scalar_t**) b_array_dev, ldb00,
                                &beta,                     c_array_dev, ldc00,
                                batch_count_00);
                        assert(status == CUBLAS_STATUS_SUCCESS);
                        a_array_dev += batch_count_00;
                        b_array_dev += batch_count_00;
                        c_array_dev += batch_count_00;
                    }

                    if (batch_count_10 > 0) {
                        cublasStatus_t status =
                            cublasGemmBatched(
                                cublas_handle,  // uses stream
                                cublas_op_const(opA), cublas_op_const(opB),
                                mb10, nb10, kb,
                                &alpha, (const scalar_t**) a_array_dev, lda10,
                                        (const scalar_t**) b_array_dev, ldb10,
                                &beta,                     c_array_dev, ldc10,
                                batch_count_10);
                        assert(status == CUBLAS_STATUS_SUCCESS);
                    }

                    error = cudaStreamSynchronize(stream);
                    assert(error == cudaSuccess);
                }

                for (int64_t j = 0; j < C.nt()-1; ++j) {
                    for (int64_t i = j+1; i < C.mt(); ++i) {  // strictly lower
                        if (C.tileIsLocal(i, j)) {
                            if (device == C.tileDevice(i, j)) {
                                // erase tmp local and remote device tiles;
                                // decrement life for remote tiles
                                A.tileErase(i, 0, device);
                                A.tileErase(j, 0, device);
                                A.tileTick(i, 0);
                                A.tileTick(j, 0);
                            }
                        }
                    }
                }
            }
            catch (std::exception& e) {
                err = __LINE__;
            }
        }
    }

    // diagonal tiles by syrk on host
    for (int64_t j = 0; j < C.nt(); ++j) {
        if (C.tileIsLocal(j, j)) {
            #pragma omp task shared(A, C, err)
            {
                try {
                    A.tileCopyToHost(j, 0, A.tileDevice(j, 0));
                    C.tileMoveToHost(j, j, C.tileDevice(j, j));
                    syrk(alpha, A(j, 0),
                         beta,  C(j, j));
                    A.tileTick(j, 0);
                    A.tileTick(j, 0);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }

    #pragma omp taskwait

    if (err) {
        throw std::exception();
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void syrk<Target::HostTask, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

template
void syrk<Target::HostNest, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

template
void syrk<Target::HostBatch, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

template
void syrk<Target::Devices, float>(
    float alpha, Matrix<float>&& A,
    float beta,  SymmetricMatrix<float>&& C,
    int priority);

// ----------------------------------------
template
void syrk<Target::HostTask, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

template
void syrk<Target::HostNest, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

template
void syrk<Target::HostBatch, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

template
void syrk<Target::Devices, double>(
    double alpha, Matrix<double>&& A,
    double beta,  SymmetricMatrix<double>&& C,
    int priority);

// ----------------------------------------
template
void syrk< Target::HostTask, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

template
void syrk< Target::HostNest, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

template
void syrk< Target::HostBatch, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

template
void syrk< Target::Devices, std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >&& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >&& C,
    int priority);

// ----------------------------------------
template
void syrk< Target::HostTask, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

template
void syrk< Target::HostNest, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

template
void syrk< Target::HostBatch, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

template
void syrk< Target::Devices, std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >&& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >&& C,
    int priority);

} // namespace internal
} // namespace slate
