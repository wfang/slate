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

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::trsm(blas::Side side, blas::Uplo uplo,
                             blas::Op op, blas::Diag diag,
                             FloatType alpha, Matrix &&a,
                                              Matrix &&b)
{
    trsm(internal::TargetType<target>(),
        side, uplo, op, diag,
        alpha, a, b);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::trsm(internal::TargetType<Target::HostTask>,
                             blas::Side side, blas::Uplo uplo,
                             blas::Op op, blas::Diag diag,
                             FloatType alpha, Matrix &a,
                                              Matrix &b)
{
    // Right, Lower, Trans
    for (int64_t m = 0; m < b.mt_; ++m)
        if (b.tileIsLocal(m, 0))
            #pragma omp task shared(a, b)
            {
                a.tileCopyToHost(0, 0, a.tileDevice(0, 0));
                b.tileMoveToHost(m, 0, b.tileDevice(m, 0));
                Tile<FloatType>::trsm(side, uplo, op, diag,
                                      alpha, a(0, 0),
                                             b(m, 0));
                a.tileTick(0, 0);
            }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template
void Matrix<double>::trsm<Target::HostTask>(
    blas::Side side, blas::Uplo uplo,
    blas::Op op, blas::Diag diag,
    double alpha, Matrix &&a,
                  Matrix &&b);

} // namespace slate
