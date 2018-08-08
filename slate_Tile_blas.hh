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

#ifndef SLATE_TILE_BLAS_HH
#define SLATE_TILE_BLAS_HH

#include <blas.hh>

#include "slate_Tile.hh"
#include "slate_util.hh"

#include <list>

namespace slate {

///=============================================================================
// Tile BLAS

///-----------------------------------------------------------------------------
/// \brief
/// General matrix multiply: $op(C) = \alpha op(A) op(B) + \beta C$.
/// Use transpose() or conj_transpose() to set $op(A)$, $op(B)$, and $op(C)$.
/// In the complex case,
/// if $op(C)$ is transpose, then $op(A)$ and $op(B)$ cannot be conj_transpose;
/// if $op(C)$ is conj_transpose, then $op(A)$ and $op(B)$ cannot be transpose.
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::gemm");

    using blas::conj;

    assert(A.uplo() == Uplo::General);
    assert(B.uplo() == Uplo::General);
    assert(C.uplo() == Uplo::General);
    assert(C.mb() == A.mb());  // m
    assert(C.nb() == B.nb());  // n
    assert(A.nb() == B.mb());  // k
    if (C.op() == Op::NoTrans) {
        // C = opA(A) opB(B) + C
        blas::gemm(blas::Layout::ColMajor,
                   A.op(), B.op(),
                   C.mb(), C.nb(), A.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
    else {
        // opC is Trans or ConjTrans
        // opC(C) = opA(A)) opB(B) + opC(C) becomes
        // C = opC(opA(A) opB(B)) + C = opC(opB(B)) opC(opA(A)) + C
        // invert opA, opB if possible; swap A <=> B; swap m <=> n
        Op opA;
        if (A.op() == Op::NoTrans)
            opA = C.op();
        else if (A.op() == C.op() || C.is_real) {
            // A and C are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        Op opB;
        if (B.op() == Op::NoTrans)
            opB = C.op();
        else if (B.op() == C.op() || C.is_real) {
            // B and C are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opB = Op::NoTrans;
        }
        else
            throw std::exception();

        if (C.op() == Op::ConjTrans) {
            alpha = conj(alpha);
            beta  = conj(beta);
        }

        blas::gemm(blas::Layout::ColMajor,
                   opB, opA,
                   C.nb(), C.mb(), A.nb(),
                   alpha, B.data(), B.stride(),
                          A.data(), A.stride(),
                   beta,  C.data(), C.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void gemm(
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    gemm(alpha, A, B, beta, C);
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian matrix multiply: $C = \alpha A op(B) + \beta op(C)$
///                         or $C = \alpha op(B) A + \beta op(C)$,
/// where $A$ is Hermitian.
/// Unlike most BLAS operations, here op(B) and op(C) must be
/// both the same, either both NoTrans or both ConjTrans.
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::hemm");

    using blas::conj;

    assert(A.mb() == A.nb());  // square
    assert(B.mb() == C.mb());
    assert(B.nb() == C.nb());
    if (side == Side::Left)
        assert(A.mb() == B.mb());
    else
        assert(A.mb() == B.nb());
    assert(B.op() == C.op());
    assert(A.op() != Op::Trans);
    assert(B.op() != Op::Trans);

    // A.op can be ignored, since A == A^T
    if (B.op() == Op::NoTrans) {
        blas::hemm(blas::Layout::ColMajor,
                   side, A.uplo(),
                   C.mb(), C.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
    else {
        // ConjTrans
        // undo transpose by swapping left <=> right, m <=> n, conj alpha & beta
        side = (side == Side::Left ? Side::Right : Side::Left);
        blas::hemm(blas::Layout::ColMajor,
                   side, A.uplo(),
                   C.nb(), C.mb(),
                   conj(alpha), A.data(), A.stride(),
                                B.data(), B.stride(),
                   conj(beta),  C.data(), C.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void hemm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    hemm(side, alpha, A, B, beta, C);
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-k update: $C = \alpha op(A) op(A)^H + \beta C$.
/// Use conj_transpose to set $op(A)$.
/// In the complex case, C cannot be transpose.
// Allowing C^T would require two conjugations: conj( conj(C) + A*A^H ).
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha, Tile<scalar_t> const& A,
    blas::real_type<scalar_t> beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::herk");

    assert(A.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    if (C.is_complex && C.op() == Op::Trans)
        throw std::exception();

    blas::herk(blas::Layout::ColMajor,
               C.uplo(), A.op(),
               C.nb(), A.nb(),
               alpha, A.data(), A.stride(),
               beta,  C.data(), C.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void herk(
    blas::real_type<scalar_t> alpha, Tile<scalar_t> const&& A,
    blas::real_type<scalar_t> beta,  Tile<scalar_t>&& C)
{
    herk(alpha, A, beta, C);
}

///-----------------------------------------------------------------------------
/// \brief
/// Hermitian rank-2k update:
///     $C = \alpha op(A) op(B)^T + \alpha op(B) op(A)^T + \beta C$.
/// Use transpose or conj_transpose to set $op(A)$ and $op(B)$.
/// In the complex case, C cannot be transpose.
// Allowing C^H would require two conjugations: conj( conj(C) + A*A^T ).
template <typename scalar_t>
void her2k(
    scalar_t alpha,                 Tile<scalar_t> const& A,
                                    Tile<scalar_t> const& B,
    blas::real_type<scalar_t> beta, Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::her2k");

    using blas::conj;

    assert(A.op() == B.op());
    assert(A.uplo() == Uplo::General);
    assert(B.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    assert(C.mb() == B.mb());  // n
    if (C.is_complex && C.op() == Op::Trans)
        throw std::exception();

    blas::her2k(blas::Layout::ColMajor,
                C.uplo(), A.op(),
                C.nb(), A.nb(),
                alpha, A.data(), A.stride(),
                       B.data(), B.stride(),
                beta,  C.data(), C.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void her2k(
    scalar_t alpha,                 Tile<scalar_t> const&& A,
                                    Tile<scalar_t> const&& B,
    blas::real_type<scalar_t> beta, Tile<scalar_t>&& C)
{
    her2k(alpha, A, B, beta, C);
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric matrix multiply: $C = \alpha A op(B) + \beta op(C)$
///                         or $C = \alpha op(B) A + \beta op(C)$,
/// where $A$ is symmetric.
/// Unlike most BLAS operations, here op(B) and op(C) must be
/// both the same, either both NoTrans or both Trans.
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::symm");

    using blas::conj;

    assert(A.mb() == A.nb());  // square
    assert(B.mb() == C.mb());
    assert(B.nb() == C.nb());
    if (side == Side::Left)
        assert(A.mb() == B.mb());
    else
        assert(A.mb() == B.nb());
    assert(B.op() == C.op());
    assert(A.op() != Op::ConjTrans);
    assert(B.op() != Op::ConjTrans);

    // A.op can be ignored, since A == A^T
    if (B.op() == Op::NoTrans) {
        blas::symm(blas::Layout::ColMajor,
                   side, A.uplo(),
                   C.mb(), C.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
    else {
        // Trans
        // undo transpose by swapping left <=> right, m <=> n
        side = (side == Side::Left ? Side::Right : Side::Left);
        blas::symm(blas::Layout::ColMajor,
                   side, A.uplo(),
                   C.nb(), C.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void symm(
    Side side,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    symm(side, alpha, A, B, beta, C);
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-k update: $C = \alpha op(A) op(A)^T + \beta C$.
/// Use transpose or conj_transpose to set $op(A)$.
/// In the complex case, C cannot be conj_transpose.
// Allowing C^H would require two conjugations: conj( conj(C) + A*A^T ).
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const& A,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::syrk");

    using blas::conj;

    assert(A.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    if (C.is_complex && C.op() == Op::ConjTrans)
        throw std::exception();

    blas::syrk(blas::Layout::ColMajor,
               C.uplo(), A.op(),
               C.nb(), A.nb(),
               alpha, A.data(), A.stride(),
               beta,  C.data(), C.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void syrk(
    scalar_t alpha, Tile<scalar_t> const&& A,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    syrk(alpha, A, beta, C);
}

///-----------------------------------------------------------------------------
/// \brief
/// Symmetric rank-2k update:
///     $C = \alpha op(A) op(B)^T + \alpha op(B) op(A)^T + \beta C$.
/// Use transpose or conj_transpose to set $op(A)$ and $op(B)$.
/// In the complex case, C cannot be conj_transpose.
// Allowing C^H would require two conjugations: conj( conj(C) + A*A^T ).
template <typename scalar_t>
void syr2k(
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t> const& B,
    scalar_t beta,  Tile<scalar_t>& C)
{
    trace::Block trace_block("blas::syr2k");

    using blas::conj;

    assert(A.op() == B.op());
    assert(A.uplo() == Uplo::General);
    assert(B.uplo() == Uplo::General);
    assert(C.mb() == C.nb());  // square
    assert(C.mb() == A.mb());  // n
    assert(C.mb() == B.mb());  // n
    if (C.is_complex && C.op() == Op::ConjTrans)
        throw std::exception();

    blas::syr2k(blas::Layout::ColMajor,
                C.uplo(), A.op(),
                C.nb(), A.nb(),
                alpha, A.data(), A.stride(),
                       B.data(), B.stride(),
                beta,  C.data(), C.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void syr2k(
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t> const&& B,
    scalar_t beta,  Tile<scalar_t>&& C)
{
    syr2k(alpha, A, B, beta, C);
}

///-----------------------------------------------------------------------------
/// \brief
template <typename scalar_t>
void trmm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B)
{
    trace::Block trace_block("blas::trmm");

    using blas::conj;

    assert(B.uplo() == Uplo::General);
    assert(A.mb() == A.nb());  // square
    assert(side == Side::Left ? A.mb() == B.mb()    // m
                              : A.mb() == B.nb());  // n
    if (B.op() == Op::NoTrans) {
        blas::trmm(blas::Layout::ColMajor,
                   side, A.uplo(), A.op(), diag,
                   B.mb(), B.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
    else {
        if (A.is_complex && A.op() != Op::NoTrans && A.op() != B.op())
            throw std::exception();

        // switch op(A) <=> op(B), side left <=> right, m <=> n
        Side side2 = (side == Side::Left ? Side::Right : Side::Left);
        Op opA;
        if (A.op() == Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || A.is_real) {
            // A and B are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans)
            alpha = conj(alpha);

        blas::trmm(blas::Layout::ColMajor,
                   side2, A.uplo(), opA, diag,
                   B.nb(), B.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void trmm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t>&& B)
{
    trmm(side, diag, alpha, A, B);
}

///-----------------------------------------------------------------------------
/// \brief
/// Triangular solve: $B = \alpha op(A)^{-1} B$ or $B = \alpha B op(A)^{-1}$.
/// Use transpose/conj_transpose to set op(A). uplo is set in the tile.
/// In the complex case,
/// if $op(B)$ is transpose, then $op(A)$ cannot be conj_transpose;
/// if $op(B)$ is conj_transpose, then $op(A)$ cannot be transpose.
template <typename scalar_t>
void trsm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const& A,
                    Tile<scalar_t>& B)
{
    trace::Block trace_block("blas::trsm");

    using blas::conj;

    assert(B.uplo() == Uplo::General);
    assert(A.mb() == A.nb());  // square
    assert(side == Side::Left ? A.mb() == B.mb()    // m
                              : A.mb() == B.nb());  // n
    if (B.op() == Op::NoTrans) {
        blas::trsm(blas::Layout::ColMajor,
                   side, A.uplo(), A.op(), diag,
                   B.mb(), B.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
    else {
        if (A.is_complex && A.op() != Op::NoTrans && A.op() != B.op())
            throw std::exception();

        // switch op(A) <=> op(B), side left <=> right, m <=> n
        Side side2 = (side == Side::Left ? Side::Right : Side::Left);
        Op opA;
        if (A.op() == Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || A.is_real) {
            // A and B are both Trans or both ConjTrans;
            // Trans == ConjTrans if real
            opA = Op::NoTrans;
        }
        else
            throw std::exception();

        if (B.op() == Op::ConjTrans)
            alpha = conj(alpha);

        blas::trsm(blas::Layout::ColMajor,
                   side2, A.uplo(), opA, diag,
                   B.nb(), B.mb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride());
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void trsm(
    Side side, Diag diag,
    scalar_t alpha, Tile<scalar_t> const&& A,
                    Tile<scalar_t>&& B)
{
    trsm(side, diag, alpha, A, B);
}

///=============================================================================
// Tile LAPACK

///-----------------------------------------------------------------------------
/// \brief
/// General matrix norm.
template <typename scalar_t>
void genorm(Norm norm, Tile<scalar_t> const& A,
            blas::real_type<scalar_t>* values)
{
    trace::Block trace_block("lapack::lange");

    assert(A.uplo() == Uplo::General);
    assert(A.op() == Op::NoTrans);

    // max norm
    // values[0] = max_{i,j} A_{i,j}
    if (norm == Norm::Max) {
        *values = lapack::lange(norm,
                                A.mb(), A.nb(),
                                A.data(), A.stride());
    }
    // one norm
    // values[j] = sum_i abs( A_{i,j} )
    else if (norm == Norm::One) {
        for (int64_t j = 0; j < A.nb(); ++j) {
            values[j] = std::abs(A(0, j));
            for (int64_t i = 1; i < A.mb(); ++i) {
                values[j] += std::abs(A(i, j));
            }
        }
    }
    // inf norm
    // values[i] = sum_j abs( A_{i,j} )
    else if (norm == Norm::Inf) {
        for (int64_t i = 0; i < A.mb(); ++i) {
            values[i] = std::abs( A(i, 0) );
        }
        for (int64_t j = 1; j < A.nb(); ++j) {
            for (int64_t i = 0; i < A.mb(); ++i) {
                values[i] += std::abs( A(i, j) );
            }
        }
    }
    // Frobenius norm
    // values[0] = scale, values[1] = sumsq such that
    // scale^2 * sumsq = sum_{i,j} abs( A_{i,j} )^2
    else if (norm == Norm::Fro) {
        values[0] = 0;  // scale
        values[1] = 1;  // sumsq
        for (int64_t j = 0; j < A.nb(); ++j) {
            lapack::lassq(A.mb(), &A.at(0, j), 1, &values[0], &values[1]);
        }
    }
    else {
        throw std::exception();  // invalid norm
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void genorm(Norm norm, Tile<scalar_t> const&& A,
            blas::real_type<scalar_t>* values)
{
    return genorm(norm, A, values);
}

///-----------------------------------------------------------------------------
/// Trapezoid and triangular matrix norm.
template <typename scalar_t>
void trnorm(Norm norm, Diag diag, Tile<scalar_t> const& A,
            blas::real_type<scalar_t>* values)
{
    using blas::max;
    using blas::min;

    trace::Block trace_block("lapack::lantr");

    assert(A.uplo() != Uplo::General);
    assert(A.op() == Op::NoTrans);

    if (norm == Norm::Max) {
        // max norm
        // values[0] = max_{i,j} A_{i,j}
        *values = lapack::lantr(norm, A.uplo(), diag,
                                A.mb(), A.nb(),
                                A.data(), A.stride());
    }
    else if (norm == Norm::One) {
        // one norm
        // values[j] = sum_i abs( A_{i,j} )
        for (int64_t j = 0; j < A.nb(); ++j) {
            values[j] = 0;
            // diagonal element
            if (j < A.mb()) {
                if (diag == Diag::Unit) {
                    values[j] += 1;
                }
                else {
                    values[j] += std::abs(A(j, j));
                }
            }
            // off-diagonal elements
            if (A.uplo() == Uplo::Lower) {
                for (int64_t i = j+1; i < A.mb(); ++i) { // strictly lower
                    values[j] += std::abs(A(i, j));
                }
            }
            else {
                for (int64_t i = 0; i < j && i < A.mb(); ++i) { // strictly upper
                    values[j] += std::abs(A(i, j));
                }
            }
        }
    }
    else if (norm == Norm::Inf) {
        // inf norm
        // values[i] = sum_j abs( A_{i,j} )
        for (int64_t i = 0; i < A.mb(); ++i) {
            values[i] = 0;
        }
        for (int64_t j = 0; j < A.nb(); ++j) {
            // diagonal element
            if (j < A.mb()) {
                if (diag == Diag::Unit) {
                    values[j] += 1;
                }
                else {
                    values[j] += std::abs(A(j, j));
                }
            }
            // off-diagonal elements
            if (A.uplo() == Uplo::Lower) {
                for (int64_t i = j+1; i < A.mb(); ++i) { // strictly lower
                    values[i] += std::abs(A(i, j));
                }
            }
            else {
                for (int64_t i = 0; i < j && i < A.mb(); ++i) { // strictly upper
                    values[i] += std::abs(A(i, j));
                }
            }
        }
    }
    else if (norm == Norm::Fro) {
        // Frobenius norm
        // values[0] = scale, values[1] = sumsq such that
        // scale^2 * sumsq = sum_{i,j} abs( A_{i,j} )^2
        values[0] = 0;  // scale
        values[1] = 1;  // sumsq
        if (diag == Diag::Unit) {
            // diagonal elements: sum 1^2 + ... + 1^2 = min( mb, nb )
            values[0] = 1;
            values[1] = min(A.mb(), A.nb());
            // off-diagonal elements
            if (A.uplo() == Uplo::Lower) {
                // strictly lower: A[ j+1:mb, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = max(A.mb() - (j+1), 0);
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(j+1, j), 1, &values[0], &values[1]);
                }
            }
            else {
                // strictly upper: A[ 0:j-1, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = min(j, A.mb());
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(0, j), 1, &values[0], &values[1]);
                }
            }
        }
        else {
            if (A.uplo() == Uplo::Lower) {
                // lower: A[ j:mb, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = max(A.mb() - j, 0);
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(j, j), 1, &values[0], &values[1]);
                }
            }
            else {
                // upper: A[ 0:j, j ]
                for (int64_t j = 0; j < A.nb(); ++j) {
                    int64_t ib = min(j+1, A.mb());
                    if (ib > 0)
                        lapack::lassq(ib, &A.at(0, j), 1, &values[0], &values[1]);
                }
            }
        }
    }
    else {
        throw std::exception();  // invalid norm
    }
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
void trnorm(Norm norm, Tile<scalar_t> const&& A,
            blas::real_type<scalar_t>* values)
{
    return trnorm(norm, A, values);
}

///-----------------------------------------------------------------------------
/// \brief
/// Cholesky factorization of tile: $L L^H = A$ or $U^H U = A$.
/// uplo is set in the tile.
template <typename scalar_t>
int64_t potrf(Tile<scalar_t>& A)
{
    trace::Block trace_block("lapack::potrf");

    return lapack::potrf(A.uplo(),
                         A.nb(),
                         A.data(), A.stride());
}

///----------------------------------------
/// Converts rvalue refs to lvalue refs.
template <typename scalar_t>
int64_t potrf(Tile<scalar_t>&& A)
{
    return potrf(A);
}

#include <unistd.h>
#include <iomanip>
#include <limits>

///-----------------------------------------------------------------------------
/// \brief
/// Swap rows of two local tiles.
///
template <typename scalar_t>
void swap(Tile<scalar_t>& A, int64_t i1,
          Tile<scalar_t>& B, int64_t i2)
{
    assert(A.nb() == B.nb());
    assert(A.op() == B.op());

    if (A.op() == Op::NoTrans) {
        blas::swap(A.nb(),
                   &A.data()[i1], A.stride(),
                   &B.data()[i2], B.stride());
    }
    else {
        // todo: op_ == Op::Trans
        assert(0);
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// Swap rows with another process.
///
template <typename scalar_t>
void swap(Tile<scalar_t>& A, int64_t i, int other_rank, MPI_Comm mpi_comm)
{
    std::vector<scalar_t> local_row(A.nb());
    std::vector<scalar_t> other_row(A.nb());

    for (int64_t j = 0; j < A.nb(); ++j)
        local_row[j] = A(i, j);

    int tag = 0;
    MPI_Sendrecv(local_row.data(), A.nb(), MPI_DOUBLE, other_rank, tag,
                 other_row.data(), A.nb(), MPI_DOUBLE, other_rank, tag,
                 mpi_comm, MPI_STATUS_IGNORE);

    for (int64_t j = 0; j < A.nb(); ++j)
         A.at(i, j) = other_row[j];
}

///-----------------------------------------------------------------------------
/// \brief
/// Compute the LU factorization of a panel.
template <typename scalar_t>
int64_t getrf(std::vector< Tile<scalar_t> >& tiles,
              std::vector<int64_t>& i_indices, std::vector<int64_t>& i_offsets,
              int thread_rank, int thread_size,
              ThreadBarrier& thread_barrier,
              std::vector<scalar_t>& max_val, std::vector<int64_t>& max_idx,
              std::vector<int64_t>& max_offs,
              scalar_t& piv_val, std::vector<scalar_t>& top_row,
              int mpi_rank, int mpi_root, MPI_Comm mpi_comm)
{
    trace::Block trace_block("lapack::getrf");

    int64_t ib = 4;
    bool root = mpi_rank == mpi_root;
    if (root)
        assert(i_indices[0] == 0);

    auto mb = tiles.at(0).mb();
    auto nb = tiles.at(0).nb();
    int64_t diag_len = std::min(nb, mb);

    // Loop over ib-wide stripes.
    for (int64_t k = 0; k < diag_len; k += ib) {

        // Loop over ib columns of a stripe.
        for (int64_t j = k; j < k+ib && j < diag_len; ++j) {

            if (root) {
                max_val[thread_rank] = tiles.at(0)(j, j);
                max_idx[thread_rank] = 0;
                max_offs[thread_rank] = j;
            }
            else {
                max_val[thread_rank] = tiles.at(0)(0, j);
                max_idx[thread_rank] = 0;
                max_offs[thread_rank] = 0;
            }

            //------------------
            // thread max search
            for (int64_t idx = thread_rank;
                 idx < tiles.size();
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = i_indices.at(idx);

                // if diagonal tile
                if (i_index == 0) {
                    for (int64_t i = j+1; i < tile.mb(); ++i) {
                        if (std::abs(tile(i, j)) >
                            std::abs(max_val[thread_rank]))
                        {
                            max_val[thread_rank] = tile(i, j);
                            max_idx[thread_rank] = idx;
                            max_offs[thread_rank] = i;
                        }
                    }

                }
                // off diagonal tiles
                else {
                    for (int64_t i = 0; i < tile.mb(); ++i) {
                        if (std::abs(tile(i, j)) >
                            std::abs(max_val[thread_rank]))
                        {
                            max_val[thread_rank] = tile(i, j);
                            max_idx[thread_rank] = idx;
                            max_offs[thread_rank] = i;
                        }
                    }
                }
            }
            thread_barrier.wait(thread_size);

            //------------------------------------
            // global max reduction and pivot swap
            if (thread_rank == 0) {

                // threads max reduction
                for (int rank = 1; rank < thread_size; ++rank) {
                    if (std::abs(max_val[rank]) > std::abs(max_val[0])) {
                        max_val[0] = max_val[rank];
                        max_idx[0] = max_idx[rank];
                        max_offs[0] = max_offs[rank];
                    }
                }

                // MPI max abs reduction
                struct { scalar_t max; int loc; } max_loc_in, max_loc;
                max_loc_in.max = std::abs(max_val[0]);
                max_loc_in.loc = mpi_rank;
                MPI_Allreduce(&max_loc_in, &max_loc, 1, MPI_DOUBLE_INT,
                              MPI_MAXLOC, mpi_comm);

                // Broadcast the pivot actual value (not abs).
                piv_val = max_val[0];
                MPI_Bcast(&piv_val, 1, MPI_DOUBLE, max_loc.loc, mpi_comm);

                // Broadcast the top row for the geru operation.
                

                //-----------
                // pivot swap

                // if I own the pivot
                if (max_loc.loc == mpi_rank) {
                    // if I am the root
                    if (root) {
                        // local swap
                        swap(tiles.at(0), j, tiles.at(max_idx[0]), max_offs[0]);
                    }
                    // I am not the root
                    else {
                        // MPI swap with the root
                        swap(tiles.at(max_idx[0]), max_offs[0], mpi_root,
                                      mpi_comm);
                    }
                }
                // I don't own the pivot
                else {
                    // I am the root
                    if (root) {
                        // MPI swap with the pivot owner
                        swap(tiles.at(0), j, max_loc.loc, mpi_comm);
                    }
                }
            }
            thread_barrier.wait(thread_size);

            // column scaling and trailing update
            for (int64_t idx = thread_rank;
                 idx < tiles.size();
                 idx += thread_size)
            {
                auto tile = tiles.at(idx);
                auto i_index = i_indices.at(idx);

                if (i_index == 0) {
                    // diagonal tile
                    for (int64_t i = j+1; i < tile.mb(); ++i)
                        tile.at(i, j) /= tile(j, j);
                }
                else {
                    // off diagonal tile
                    for (int64_t i = 0; i < tile.mb(); ++i)
                        tile.at(i, j) /= piv_val;
                }

                // todo: make it a tile operation
                if (i_index == 0) {
                    blas::geru(blas::Layout::ColMajor,
                               tile.mb()-j-1, std::min(k+ib-j-1, diag_len-j-1),
                               -1.0, &tile.data()[j+1+j*tile.stride()], 1,
                                     &tile.data()[j+(j+1)*tile.stride()], tile.stride(),
                                     &tile.data()[j+1+(j+1)*tile.stride()], tile.stride());
                }
                else {
                    // blas::geru(blas::Layout::ColMajor,
                    //            tile.mb(), std::min(k+ib-j-1, diag_len-j-1),
                    //            -1.0, &tile.data()[j*tile.stride()], 1,
                    //                  &tile.data()[j+(j+1)*tile.stride()], tile.stride(),
                    //                  &tile.data()[j+1+(j+1)*tile.stride()], tile.stride());
                }


            }
            thread_barrier.wait(thread_size);


        }


    }


}

} // namespace slate

#endif // SLATE_TILE_BLAS_HH
