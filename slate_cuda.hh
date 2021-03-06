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
#ifndef SLATE_CUDA_HH
#define SLATE_CUDA_HH

#if defined(SLATE_WITH_CUDA) || defined(__NVCC__)
    #include <cuda_runtime.h>
    #include <cuComplex.h>
#else

#include <cstdlib>
#include <complex>

typedef int cuComplex;
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef int cudaMemcpyKind;
typedef std::complex<float> cuFloatComplex;
typedef std::complex<double> cuDoubleComplex;

enum {
    cudaMemcpyDeviceToHost,
    cudaMemcpyHostToDevice,
    cudaStreamNonBlocking,
    cudaSuccess
};

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceCount(int* count);

cudaError_t cudaMalloc(void** devPtr, size_t size);

cudaError_t cudaMallocHost(void** ptr, size_t size);

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                              const void* src, size_t spitch,
                              size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream = 0);

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0);

cudaError_t cudaMemcpy(void* dst, const void*  src,
                       size_t count, cudaMemcpyKind kind);

cudaError_t cudaSetDevice(int device);

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamDestroy(cudaStream_t pStream);

char* cudaGetErrorName(cudaError_t error);
char* cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif // not SLATE_WITH_CUDA

#endif // SLATE_CUDA_HH
