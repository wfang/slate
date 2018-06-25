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

#include "slate_device.hh"

#include <cstdio>
#include <cuComplex.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// max that propogates nan consistently:
///     max_nan( 1,   nan ) = nan
///     max_nan( nan, 1   ) = nan
///
/// For x=nan, y=1:
/// nan < y is false, yields x (nan)
///
/// For x=1, y=nan:
/// x < nan    is false, would yield x, but
/// isnan(nan) is true, yields y (nan)
///
template <typename T>
__host__ __device__
inline T max_nan(T x, T y)
{
    return (isnan(y) || (x) < (y) ? (y) : (x));
}

//------------------------------------------------------------------------------
/// Max reduction of n-element array x, leaving total in x[0]. Propogates NaN
/// values consistently.
/// With k threads, can reduce array up to 2*k in size. Assumes number of
/// threads <= 1024, which is the current max number of CUDA threads.
///
/// @param[in] n
///		Size of array.
///
/// @param[in] tid
///     Thread id.
///
/// @param[in] x
///     Array of dimension n. On exit, x[0] = max(x[0], ..., x[n-1]);
///     the rest of x is overwritten.
///
template <typename T>
__device__ void
max_nan_reduce(int n, int tid, T* x)
{
    if (n > 1024) { if (tid < 1024 && tid + 1024 < n) { x[tid] = max_nan(x[tid], x[tid+1024]); }  __syncthreads(); }
    if (n >  512) { if (tid <  512 && tid +  512 < n) { x[tid] = max_nan(x[tid], x[tid+ 512]); }  __syncthreads(); }
    if (n >  256) { if (tid <  256 && tid +  256 < n) { x[tid] = max_nan(x[tid], x[tid+ 256]); }  __syncthreads(); }
    if (n >  128) { if (tid <  128 && tid +  128 < n) { x[tid] = max_nan(x[tid], x[tid+ 128]); }  __syncthreads(); }
    if (n >   64) { if (tid <   64 && tid +   64 < n) { x[tid] = max_nan(x[tid], x[tid+  64]); }  __syncthreads(); }
    if (n >   32) { if (tid <   32 && tid +   32 < n) { x[tid] = max_nan(x[tid], x[tid+  32]); }  __syncthreads(); }
    if (n >   16) { if (tid <   16 && tid +   16 < n) { x[tid] = max_nan(x[tid], x[tid+  16]); }  __syncthreads(); }
    if (n >    8) { if (tid <    8 && tid +    8 < n) { x[tid] = max_nan(x[tid], x[tid+   8]); }  __syncthreads(); }
    if (n >    4) { if (tid <    4 && tid +    4 < n) { x[tid] = max_nan(x[tid], x[tid+   4]); }  __syncthreads(); }
    if (n >    2) { if (tid <    2 && tid +    2 < n) { x[tid] = max_nan(x[tid], x[tid+   2]); }  __syncthreads(); }
    if (n >    1) { if (tid <    1 && tid +    1 < n) { x[tid] = max_nan(x[tid], x[tid+   1]); }  __syncthreads(); }
}

//------------------------------------------------------------------------------
/// Sum reduction of n-element array x, leaving total in x[0].
/// With k threads, can reduce array up to 2*k in size. Assumes number of
/// threads <= 1024 (which is current max number of CUDA threads).
///
/// @param[in] n
///		Size of array.
///
/// @param[in] tid
///     Thread id.
///
/// @param[in] x
///     Array of dimension n. On exit, x[0] = sum(x[0], ..., x[n-1]);
///     rest of x is overwritten.
///
template <typename T>
__device__ void
sum_reduce(int n, int tid, T* x)
{
    if (n > 1024) { if (tid < 1024 && tid + 1024 < n) { x[tid] += x[tid+1024]; }  __syncthreads(); }
    if (n >  512) { if (tid <  512 && tid +  512 < n) { x[tid] += x[tid+ 512]; }  __syncthreads(); }
    if (n >  256) { if (tid <  256 && tid +  256 < n) { x[tid] += x[tid+ 256]; }  __syncthreads(); }
    if (n >  128) { if (tid <  128 && tid +  128 < n) { x[tid] += x[tid+ 128]; }  __syncthreads(); }
    if (n >   64) { if (tid <   64 && tid +   64 < n) { x[tid] += x[tid+  64]; }  __syncthreads(); }
    if (n >   32) { if (tid <   32 && tid +   32 < n) { x[tid] += x[tid+  32]; }  __syncthreads(); }
    if (n >   16) { if (tid <   16 && tid +   16 < n) { x[tid] += x[tid+  16]; }  __syncthreads(); }
    if (n >    8) { if (tid <    8 && tid +    8 < n) { x[tid] += x[tid+   8]; }  __syncthreads(); }
    if (n >    4) { if (tid <    4 && tid +    4 < n) { x[tid] += x[tid+   4]; }  __syncthreads(); }
    if (n >    2) { if (tid <    2 && tid +    2 < n) { x[tid] += x[tid+   2]; }  __syncthreads(); }
    if (n >    1) { if (tid <    1 && tid +    1 < n) { x[tid] += x[tid+   1]; }  __syncthreads(); }
}

//------------------------------------------------------------------------------
/// Overloaded versions of absolute value on device.
__host__ __device__
inline float abs(float x)
{
    return fabsf(x);
}

__host__ __device__
inline double abs(double x)
{
    return fabs(x);
}

__host__ __device__
inline float abs(cuFloatComplex x)
{
    return cuCabsf(x);
}

__host__ __device__
inline double abs(cuDoubleComplex x)
{
    return cuCabs(x);
}

///-----------------------------------------------------------------------------
/// Square of number.
/// @return x^2
template <typename scalar_t>
__host__ __device__
inline scalar_t sqr(scalar_t x)
{
    return x*x;
}

//------------------------------------------------------------------------------
/// Adds two scaled, sum-of-squares representations.
/// On exit, scale1 and sumsq1 are updated such that:
///     scale1^2 sumsq1 := scale1^2 sumsq1 + scale2^2 sumsq2.
template <typename real_t>
__host__ __device__
void add_sumsq(
    real_t&       scale1, real_t&       sumsq1,
    real_t const& scale2, real_t const& sumsq2 )
{
    if (scale1 > scale2) {
        sumsq1 = sumsq1 + sumsq2*sqr(scale2 / scale1);
        // scale1 stays same
    }
    else {
        sumsq1 = sumsq1*sqr(scale1 / scale2) + sumsq2;
        scale1 = scale2;
    }
}

//------------------------------------------------------------------------------
/// Adds new value to scaled, sum-of-squares representation.
/// On exit, scale and sumsq are updated such that:
///     scale^2 sumsq := scale^2 sumsq + (absx)^2
template <typename real_t>
__host__ __device__
inline void add_sumsq(
    real_t& scale, real_t& sumsq,
    real_t absx)
{
    if (scale < absx) {
        sumsq = 1 + sumsq * sqr(scale / absx);
        scale = absx;
    }
    else {
        sumsq = sumsq + sqr(absx / scale);
    }
}

//------------------------------------------------------------------------------
/// Finds the largest absolute value of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * n.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
template <typename scalar_t>
__global__ void synormMaxKernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    scalar_t const* row = &tile[threadIdx.x];

    // Each thread finds max of one row.
    // This does coalesced reads of one column at a time in parallel.
    real_t max = 0;
    if (uplo == lapack::Uplo::Lower) {
        for (int64_t j = 0; j <= threadIdx.x && j < n; ++j) // lower
            max = max_nan(max, abs(row[j*lda]));
    }
    else {
        // Loop backwards (n-1 down to i) to maintain coalesced reads.
        for (int64_t j = n-1; j >= threadIdx.x; --j) // upper
            max = max_nan(max, abs(row[j*lda]));
    }

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_max = (real_t*) dynamic_data;
    row_max[threadIdx.x] = max;
    __syncthreads();

    // Reduction to find max of tile.
    max_nan_reduce(blockDim.x, threadIdx.x, row_max);
    if (threadIdx.x == 0) {
        tiles_maxima[blockIdx.x] = row_max[0];
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
__global__ void synormOneKernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    scalar_t const* row    = &tile[threadIdx.x];
    scalar_t const* column = &tile[lda*threadIdx.x];

    // Each thread sums one row/column.
    // todo: the row reads are coalesced, but the col reads are not coalesced
    real_t sum = 0;
    if (uplo == lapack::Uplo::Lower) {
        for (int64_t j = 0; j <= threadIdx.x; ++j) // lower
            sum += abs(row[j*lda]);
        for (int64_t i = threadIdx.x + 1; i < n; ++i) // strictly lower
            sum += abs(column[i]);
    }
    else {
        // Loop backwards (n-1 down to i) to maintain coalesced reads.
        for (int64_t j = n-1; j >= threadIdx.x; --j) // upper
            sum += abs(row[j*lda]);
        for (int64_t i = 0; i < threadIdx.x && i < n; ++i) // strictly upper
            sum += abs(column[i]);
    }

    tiles_sums[blockIdx.x*ldv + threadIdx.x] = sum;
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension blockDim.x,
///     where each tiles[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
__global__ void synormFroKernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* tiles_values)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];
    scalar_t const* row = &tile[threadIdx.x];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
    real_t scale = 0;
    real_t sumsq = 1;
    if (uplo == lapack::Uplo::Lower) {
        for (int64_t j = 0; j < threadIdx.x && j < n; ++j) // strictly lower
            add_sumsq(scale, sumsq, abs(row[j*lda]));
        // double for symmetric entries
        sumsq *= 2;
        // diagonal
        add_sumsq(scale, sumsq, abs(row[threadIdx.x*lda]));
    }
    else {
        // Loop backwards (n-1 down to i) to maintain coalesced reads.
        for (int64_t j = n-1; j > threadIdx.x; --j) // strictly upper
            add_sumsq(scale, sumsq, abs(row[j*lda]));
        // double for symmetric entries
        sumsq *= 2;
        // diagonal
        add_sumsq(scale, sumsq, abs(row[threadIdx.x*lda]));
    }

    // Save partial results in shared memory.
    extern __shared__ char dynamic_data[];
    real_t* row_scale = (real_t*) &dynamic_data[0];
    real_t* row_sumsq = &row_scale[n];
    row_scale[threadIdx.x] = scale;
    row_sumsq[threadIdx.x] = sumsq;
    __syncthreads();

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    if (threadIdx.x == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
        for (int64_t i = 1; i < n; ++i)
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
/// i = 0, ..., n-1,
/// j = 0, ..., n-1.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///     Currently, n <= 1024 due to CUDA implementation.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
///     - Norm::Inf: for symmetric, same as Norm::One
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
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
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
        if (n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count, stream);
        }
        else {
            assert(n <= 1024);
            assert(ldv == 1);
            // Max 1024 threads * 8 bytes = 8 KiB shared memory in double [complex].
            synormMaxKernel<<<batch_count, n, sizeof(real_t) * n, stream>>>
                (uplo, n, Aarray, lda, values);
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        if (n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * n, stream);
        }
        else {
            assert(n <= 1024);
            assert(ldv >= n);
            synormOneKernel<<<batch_count, n, 0, stream>>>
                (uplo, n, Aarray, lda, values, ldv);
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (n == 0) {
            cudaMemsetAsync(values, 0, sizeof(real_t) * batch_count * 2, stream);
        }
        else {
            assert(n <= 1024);
            assert(ldv == 2);
            // Max 1024 threads * 16 bytes = 16 KiB shared memory in double [complex].
            synormFroKernel<<<batch_count, n, sizeof(real_t) * n * 2, stream>>>
                (uplo, n, Aarray, lda, values);
        }
    }

    // check that launch succeeded (could still have async errors)
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::exception();
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each row and each column of elements,
/// for each tile in tiles.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by synorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///     n <= 1024 for current CUDA architectures (2.x to 6.x).
///
/// @param[in] tiles
///     Array of tiles of dimension gridDim.x,
///     where each tiles[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
__global__ void synormOffdiagOneKernel(
    int64_t m, int64_t n,
    scalar_t const* const* tiles, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const* tile = tiles[blockIdx.x];

    // todo: use blocking to read the tile just once.
    if (threadIdx.x < m) {
        // Each thread sums one row.
        // This does coalesced reads of one column at a time in parallel.
        scalar_t const* row = &tile[threadIdx.x];
        real_t sum = 0;
        for (int64_t j = 0; j < n; ++j)
            sum += abs(row[j*lda]);

        values[blockIdx.x*ldv + n + threadIdx.x] = sum;
    }

    if (threadIdx.x < n) {
        // Each thread sums one column.
        // todo: reads are not coalesced
        scalar_t const* column = &tile[lda*threadIdx.x];
        real_t sum = 0;
        for (int64_t i = 0; i < m; ++i)
            sum += abs(column[i]);

        values[blockIdx.x*ldv + threadIdx.x] = sum;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that returns the largest absolute value of elements for
/// each tile in Aarray. Sets
///     tiles_maxima[k] = max_{i, j}( abs( A^(k)_(i, j) )),
/// for each tile A^(k), where
/// A^(k) = Aarray[k],
/// k = 0, ..., blockDim.x-1,
/// i = 0, ..., n-1,
/// j = 0, ..., n-1.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///     Currently, n <= 1024 due to CUDA implementation.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
///     - Norm::Inf: for symmetric, same as Norm::One
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
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream)
{
    // quick return
    if (batch_count == 0)
        return;

    //---------
    // one norm
    if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        assert(m <= 1024);
        assert(n <= 1024);
        assert(ldv >= n);
        int nthreads = std::max(m, n);
        synormOffdiagOneKernel<<<batch_count, nthreads, 0, stream>>>
            (m, n, Aarray, lda, values, ldv);
    }
    else {
        throw std::exception();
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
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    cuFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

template
void synorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    cuDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    cudaStream_t stream);

//----------------------------------------
template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    cuFloatComplex const* const* Aarray, int64_t lda,
    float* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

template
void synormOffdiag(
    lapack::Norm norm,
    int64_t m, int64_t n,
    cuDoubleComplex const* const* Aarray, int64_t lda,
    double* values, int64_t ldv,
    int64_t batch_count,
    cudaStream_t stream);

} // namespace device
} // namespace slate
