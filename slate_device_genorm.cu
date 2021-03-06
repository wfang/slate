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

#include "slate_device.hh"
#include "slate_device_util.cuh"

#include <cstdio>
#include <cuComplex.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Finds the largest absolute value of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * m.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     m <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
template <typename scalar_t>
__global__ void genormMaxKernel(
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    const int idx = threadIdx.x;
    scalar_t const* row = &tile[idx];

    // Each thread finds max of one row.
    // This does coalesced reads of one column at a time in parallel.
    real_t max = abs(row[0]);
    for (int64_t j = 1; j < n; ++j)
        max = max_nan(max, abs(row[j*lda]));

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_max = (real_t*) dynamic_data;
    row_max[idx] = max;
    __syncthreads();

    // Reduction to find max of tile.
    max_nan_reduce(blockDim.x, idx, row_max);
    if (idx == 0) {
        tiles_maxima[blockIdx.x] = row_max[0];
    }
}

const int one_ib = 32;
const int one_ib1 = 33;

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + j] = max_{i} abs( A^(k)_(i, j) )
///     for row j of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
__global__ void genormOneKernel(
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    extern __shared__ char dynamic_data[];
    real_t* shmem_tile = (real_t*)dynamic_data;
    const int idx = threadIdx.x;

    for (int64_t jj = 0; jj < n; jj += one_ib) {
        real_t sum = 0.0;
        for (int64_t ii = 0; ii < m; ii += one_ib) {
            // Read 32x32 sub-tile into shared memory.
            // This does coalesced reads of one column at a time in parallel.
            for (int64_t j = 0; j < one_ib; ++j)
                if (jj+j < n && ii+idx < m)
                    shmem_tile[j*one_ib1 + idx] = abs(tile[(jj+j)*lda + ii+idx]);
            __syncthreads();  // shmem_tile loaded

            // Each thread sums one column.
            for (int64_t i = 0; i < one_ib; ++i)
                if (jj+idx < n && ii+i < m)
                    sum += shmem_tile[idx*one_ib1 + i];
            __syncthreads();  // done with shmem_tile
        }

        if (jj+idx < n)
            tiles_sums[blockIdx.x*ldv + jj+idx] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each row of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block, hence,
///     m <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///     for row i of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
__global__ void genormInfKernel(
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    const int idx = threadIdx.x;
    scalar_t const* row = &tile[idx];

    // Each thread sums one row.
    // This does coalesced reads of one column at a time in parallel.
    real_t sum = abs(row[0]);
    for (int64_t j = 1; j < n; ++j)
        sum += abs(row[j*lda]);

    tiles_sums[blockIdx.x*ldv + idx] = sum;
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension blockDim.x,
///     where each tiles[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_values
///     Array of dimension 2 * blockDim.x.
///     On exit,
///         tiles_values[2*k + 0] = scale
///         tiles_values[2*k + 1] = sumsq
///     such that scale^2 * sumsq = sum_{i,j} abs( A^(k)_{i,j} )^2
///     for tile A^(k).
///
template <typename scalar_t>
__global__ void genormFroKernel(
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_values)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    const int idx = threadIdx.x;
    scalar_t const* row = &tile[idx];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
    real_t scale = abs(row[0]);
    real_t sumsq = 1;
    for (int64_t j = 1; j < n; ++j) {
        add_sumsq(scale, sumsq, abs(row[j*lda]));
    }

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_scale = (real_t*) &dynamic_data[0];
    real_t* row_sumsq = &row_scale[m];
    row_scale[idx] = scale;
    row_sumsq[idx] = sumsq;
    __syncthreads();

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    if (idx == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
        for (int64_t i = 1; i < m; ++i)
            add_sumsq(tile_scale, tile_sumsq, row_scale[i], row_sumsq[i]);

        tiles_values[blockIdx.x*2 + 0] = tile_scale;
        tiles_values[blockIdx.x*2 + 1] = tile_sumsq;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that returns the largest absolute value of elements for
/// each tile in Aarray. Sets
///     tiles_maxima[k] = max_{i, j}( abs( A^(k)_(i, j) )),
/// for each tile A^(k), where
/// A^(k) = Aarray[k],
/// k = 0, ..., blockDim.x-1,
/// i = 0, ..., m-1,
/// j = 0, ..., n-1.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///     Currently, n <= 1024 due to CUDA implementation.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] values
///     Array in GPU memory, dimension batch_count * ldv.
///     - Norm::Max: ldv = 1.
///         On exit, values[k] = max_{i, j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count.
///
///     - Norm::One: ldv >= n.
///         On exit, values[k*ldv + j] = sum_{i} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= j < n.
///
///     - Norm::Inf: ldv >= m.
///         On exit, values[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= i < m.
///
///     - Norm::Max: ldv = 2.
///         On exit,
///             values[k*2 + 0] = scale_k
///             values[k*2 + 1] = sumsq_k
///         where scale_k^2 sumsq_k = sum_{i,j} abs( A^(k)_(i, j) )^2
///         for 0 <= k < batch_count.
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] stream
///     CUDA stream to execute in.
///
template <typename scalar_t>
void genorm(
    lapack::Norm norm,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream)
{
    using real_t = blas::real_type<scalar_t>;

    // quick return
    if (batch_count == 0)
        return;

    //---------
    // max norm
    if (norm == lapack::Norm::Max) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count, stream);
        }
        else {
            assert(m <= 1024);
            assert(ldv == 1);
            // Max 1024 threads * 8 bytes = 8 KiB shared memory in double [complex].
            genormMaxKernel<<<batch_count, m, sizeof(real_t) * m, stream>>>
                (m, n, Aarray, lda, values);
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * n, stream);
        }
        else {
            assert(ldv >= n);
            genormOneKernel<<<batch_count, one_ib, sizeof(real_t)*one_ib*one_ib1, stream>>>
                (m, n, Aarray, lda, values, ldv);
        }
    }
    //---------
    // inf norm
    else if (norm == lapack::Norm::Inf) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * m, stream);
        }
        else {
            assert(m <= 1024);
            assert(ldv >= m);
            genormInfKernel<<<batch_count, m, 0, stream>>>
                (m, n, Aarray, lda, values, ldv);
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (m == 0 || n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * 2, stream);
        }
        else {
            assert(m <= 1024);
            assert(ldv == 2);
            // Max 1024 threads * 16 bytes = 16 KiB shared memory in double [complex].
            genormFroKernel<<<batch_count, m, sizeof(real_t) * m * 2, stream>>>
                (m, n, Aarray, lda, values);
        }
    }

    // check that launch succeeded (could still have async errors)
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::exception();
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void genorm(
    lapack::Norm norm,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void genorm(
    lapack::Norm norm,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void genorm(
    lapack::Norm norm,
    int64_t m, int64_t n,
    cuFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void genorm(
    lapack::Norm norm,
    int64_t m, int64_t n,
    cuDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

} // namespace device
} // namespace slate
