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

#ifndef SLATE_MEMORY_HH
#define SLATE_MEMORY_HH

#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <map>
#include <stack>

#include "slate_cuda.hh"
#include "slate_openmp.hh"

namespace slate {

///-----------------------------------------------------------------------------
/// \class
/// \brief
/// Allocates workspace blocks for host and GPU devices.
/// Currently assumes a fixed-size block of block_size bytes,
/// e.g., block_size = sizeof(scalar_t) * nb * nb.
class Memory {
public:
    friend class Debug;

    Memory(size_t block_size);
    ~Memory();

    // todo: change add* to reserve*?
    void addHostBlocks(int64_t num_blocks);
    void addDeviceBlocks(int device, int64_t num_blocks);

    void clearHostBlocks();
    void clearDeviceBlocks(int device);

    void* alloc(int device);
    void free(void *block, int device);

    /// @return number of available free blocks in device's memory pool,
    /// which can be host.
    size_t available(int device) const
    {
        return free_blocks_.at(device).size();
    }

    /// @return total number of blocks in device's memory pool,
    /// which can be host.
    size_t capacity(int device) const
    {
        return capacity_.at(device);
    }

    // ----------------------------------------
    // public static variables
    static int host_num_;
    static int num_devices_;

private:
    void* allocBlock(int device);

    void* allocHostMemory(size_t size);
    void* allocDeviceMemory(int device, size_t size);

    void freeHostMemory(void *host_mem);
    void freeDeviceMemory(int device, void *dev_mem);

    // ----------------------------------------
    // member variables
    size_t block_size_;

    // map device number to stack of blocks
    std::map< int, std::stack< void* > > free_blocks_;
    std::map< int, std::stack< void* > > allocated_mem_;
    std::map< int, size_t > capacity_;
};

} // namespace slate

#endif // SLATE_MEMORY_HH
