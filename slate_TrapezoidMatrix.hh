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

#ifndef SLATE_TRAPEZOID_MATRIX_HH
#define SLATE_TRAPEZOID_MATRIX_HH

#include "slate_BaseTrapezoidMatrix.hh"
#include "slate_Matrix.hh"
#include "slate_Tile.hh"
#include "slate_types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate_mpi.hh"

namespace slate {

//==============================================================================
/// Symmetric, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class TrapezoidMatrix: public BaseTrapezoidMatrix<scalar_t> {
public:
    // constructors
    TrapezoidMatrix();

    TrapezoidMatrix(Uplo uplo, Diag diag, int64_t m, int64_t n, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm);

    static
    TrapezoidMatrix fromLAPACK(Uplo uplo, Diag diag, int64_t m, int64_t n,
                               scalar_t* A, int64_t lda, int64_t nb,
                               int p, int q, MPI_Comm mpi_comm);

    static
    TrapezoidMatrix fromScaLAPACK(Uplo uplo, Diag diag, int64_t m, int64_t n,
                                  scalar_t* A, int64_t lda, int64_t nb,
                                  int p, int q, MPI_Comm mpi_comm);

    static
    TrapezoidMatrix fromDevices(Uplo uplo, Diag diag, int64_t m, int64_t n,
                                scalar_t** A, int num_devices, int64_t lda,
                                int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // conversion
    TrapezoidMatrix(Diag diag, BaseTrapezoidMatrix< scalar_t >& orig);

    TrapezoidMatrix(Uplo uplo, Diag diag, Matrix<scalar_t>& orig);

    // conversion sub-matrix
    TrapezoidMatrix(Diag diag, BaseTrapezoidMatrix<scalar_t>& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2);

    TrapezoidMatrix(Uplo uplo, Diag diag, Matrix<scalar_t>& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2);

    // sub-matrix
    TrapezoidMatrix sub(int64_t i1, int64_t i2);

    Matrix<scalar_t> sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2);

protected:
    // used by fromLAPACK
    TrapezoidMatrix(Uplo uplo, Diag diag, int64_t m, int64_t n,
                    scalar_t* A, int64_t lda, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm);

    // used by fromScaLAPACK
    TrapezoidMatrix(Uplo uplo, Diag diag, int64_t m, int64_t n,
                    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm);

    // used by fromDevices
    TrapezoidMatrix(Uplo uplo, Diag diag, int64_t m, int64_t n,
                    scalar_t** Aarray, int num_devices, int64_t lda, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm);

    // used by sub
    TrapezoidMatrix(TrapezoidMatrix& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2);

public:
    template <typename T>
    friend void swap(TrapezoidMatrix<T>& A, TrapezoidMatrix<T>& B);

    Diag diag() { return diag_; }
    void diag(Diag in_diag) { diag_ = in_diag; }

protected:
    Diag diag_;
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix()
    : BaseTrapezoidMatrix<scalar_t>()
{}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n matrix, with no tiles allocated.
/// Tiles can be added with tileInsert().
//
// todo: have allocate flag? If true, allocate data; else user will insert tiles?
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Uplo uplo, Diag diag, int64_t m, int64_t n, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTrapezoidMatrix<scalar_t>(uplo, m, n, nb, p, q, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from LAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper symmetric LAPACK-style matrix.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The n-by-n symmetric matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of the array A. lda >= m.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t> TrapezoidMatrix<scalar_t>::fromLAPACK(
    Uplo uplo, Diag diag, int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    return TrapezoidMatrix<scalar_t>(uplo, diag, m, n, A, lda, nb,
                                     p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from ScaLAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper symmetric ScaLAPACK-style matrix.
/// @see BaseTrapezoidMatrix
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The local portion of the 2D block cyclic distribution of
///     the n-by-n matrix A, with local leading dimension lda.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t> TrapezoidMatrix<scalar_t>::fromScaLAPACK(
    Uplo uplo, Diag diag, int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    // note extra nb
    return TrapezoidMatrix<scalar_t>(uplo, diag, m, n, A, lda, nb, nb,
                                     p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from ScaLAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper symmetric ScaLAPACK-style matrix.
/// @see BaseTrapezoidMatrix
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The local portion of the 2D block cyclic distribution of
///     the n-by-n matrix A, with local leading dimension lda.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t> TrapezoidMatrix<scalar_t>::fromDevices(
    Uplo uplo, Diag diag, int64_t m, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    return TrapezoidMatrix<scalar_t>(uplo, diag, m, n, Aarray, num_devices, lda, nb,
                                     p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// @see fromLAPACK
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Uplo uplo, Diag diag, int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTrapezoidMatrix<scalar_t>(uplo, m, n, A, lda, nb, p, q, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// @see fromScaLAPACK
/// This differs from LAPACK constructor by adding mb.
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Uplo uplo, Diag diag, int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTrapezoidMatrix<scalar_t>(uplo, m, n, A, lda, mb, nb, p, q, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// @see fromDevices
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Uplo uplo, Diag diag, int64_t m, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTrapezoidMatrix<scalar_t>(uplo, m, n, Aarray, num_devices, lda, nb,
                                    p, q, mpi_comm),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Diag diag, BaseTrapezoidMatrix<scalar_t>& orig)
    : BaseTrapezoidMatrix<scalar_t>(orig),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
/// creates a shallow copy view of the original matrix, A[ i1:i2, j1:j2 ].
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Diag diag, BaseTrapezoidMatrix<scalar_t>& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseTrapezoidMatrix<scalar_t>(orig, i1, i2, j1, j2),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// Conversion from general matrix
/// creates a shallow copy view of the original matrix.
///
/// @param[in] in_uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Uplo uplo, Diag diag, Matrix<scalar_t>& orig)
    : BaseTrapezoidMatrix<scalar_t>(uplo, orig),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// Conversion from general matrix, sub-matrix constructor
/// creates shallow copy view of original matrix, A[ i1:i2, j1:j2 ].
///
/// @param[in] diag
///     - NonUnit: A does not have unit diagonal.
///     - Unit:    A has unit diagonal; diagonal elements are not referenced
///                and are assumed to be one.
///
/// @param[in,out] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    Uplo uplo, Diag diag, Matrix<scalar_t>& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseTrapezoidMatrix<scalar_t>(uplo, orig, i1, i2, j1, j2),
      diag_(diag)
{}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, j1:j2 ]. Requires i1 == j1. The new view is still a trapezoid
/// matrix, with the same diagonal as the parent matrix.
///
/// @param[in,out] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t>::TrapezoidMatrix(
    TrapezoidMatrix& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseTrapezoidMatrix<scalar_t>(orig, i1, i2, j1, j2),
      diag_(orig.diag())
{}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, i1:i2 ].
/// This version returns a TrapezoidMatrix with the same diagonal as the
/// parent matrix.
/// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
///                                  int64_t j1, int64_t j2)
///
/// @param[in] i1
///     Starting block row and column index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row and column index (inclusive). i2 < mt.
///
template <typename scalar_t>
TrapezoidMatrix<scalar_t> TrapezoidMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2)
{
    return TrapezoidMatrix(*this, i1, i2, i1, i2);
}

//------------------------------------------------------------------------------
/// Returns off-diagonal sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
/// This version returns a general Matrix, which:
/// - if uplo = Lower, is strictly below the diagonal, or
/// - if uplo = Upper, is strictly above the diagonal.
/// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///
template <typename scalar_t>
Matrix<scalar_t> TrapezoidMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    return BaseTrapezoidMatrix<scalar_t>::sub(i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Swaps contents of matrices A and B.
template <typename scalar_t>
void swap(TrapezoidMatrix<scalar_t>& A, TrapezoidMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseTrapezoidMatrix<scalar_t>& >(A),
         static_cast< BaseTrapezoidMatrix<scalar_t>& >(B));
    swap(A.diag_, B.diag_);
}

} // namespace slate

#endif // SLATE_TRAPEZOID_MATRIX_HH
