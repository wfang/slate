
#ifndef SLATE_TILE_HH
#define SLATE_TILE_HH

#include "slate_Memory.hh"

#include "blas.hh"
#include "lapack.hh"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifdef SLATE_WITH_CUDA
    #include <cuda_runtime.h>
#else
    #include "slate_NoCuda.hh"
#endif

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#include <omp.h>

extern "C" void trace_cpu_start();
extern "C" void trace_cpu_stop(const char *color);

namespace slate {

//------------------------------------------------------------------------------
template<typename FloatType>
class Tile {
public:
    int64_t mb_;
    int64_t nb_;

    FloatType *data_;
    bool local_ = true;
    int64_t life_;

    MPI_Request bcast_request_;
    MPI_Group bcast_group_;
    MPI_Comm bcast_comm_;

    static int host_num_;
    int device_num_;

    static Memory memory_;

    //-----------------------------------
    size_t size() {
        return sizeof(FloatType)*mb_*nb_;
    }
    void allocate()
    {
//      trace_cpu_start();
        // data_ = (FloatType*)omp_target_alloc(size(), device_num_);
        // assert(data_ != nullptr);
        if (device_num_ == host_num_) {
            // data_ = (FloatType*)malloc(size());
            // assert(data_ != nullptr);
            // cudaError_t error = cudaMallocHost(&data_, size());
            // assert(error == cudaSuccess);
            data_ = (FloatType*)memory_.alloc_host();
        }
        else {
            cudaError_t error;
            error = cudaSetDevice(device_num_);
            assert(error == cudaSuccess);
            // error = cudaMalloc(&data_, size());
            // assert(error == cudaSuccess);
            data_ = (FloatType*)memory_.alloc();
        }
//      trace_cpu_stop("Orchid");
    }
    void deallocate()
    {
//      trace_cpu_start();
        // omp_target_free(data_, device_num_);
        if (device_num_ == host_num_) {
            // free(data_);
            // cudaFree(data_);
            memory_.free_host(data_);
        }
        else {
            cudaError_t error = cudaSetDevice(device_num_);
            assert(error == cudaSuccess);
            // cudaFree(data_);
            memory_.free(data_);
        }
        data_ = nullptr;
//      trace_cpu_stop("Crimson");
    }
    void copyTo(FloatType *a, int64_t lda)
    {
        for (int64_t n = 0; n < nb_; ++n)
            memcpy(&data_[n*mb_], &a[n*lda], sizeof(FloatType)*mb_);
    }
    void copyFrom(FloatType *a, int64_t lda)
    {
        for (int64_t n = 0; n < nb_; ++n)
            memcpy(&a[n*lda], &data_[n*mb_], sizeof(FloatType)*mb_);
    }

    void tick()
    {
        if (!this->local_)
            #pragma omp critical
            {
                --this->life_;
                if (this->life_ == 0)
                    this->deallocate();
            }
    }

    void tick(Tile<FloatType> *tile)
    {
        if (!tile->local_)
            #pragma omp critical
            {
                --tile->life_;
                if (tile->life_ == 0)
                    tile->deallocate();
            }
    }
    Tile()
    {
	printf("Tile(): this should not happen!\n");
	assert(0);
    }
    Tile(int64_t mb, int64_t nb)
        : mb_(mb), nb_(nb), device_num_(host_num_), life_(0)
    {
        allocate();
    }
    Tile(int64_t mb, int64_t nb, FloatType *a, int64_t lda)
        : mb_(mb), nb_(nb), device_num_(host_num_), life_(0)
    {
        allocate();
        copyTo(a, lda);
    }
    Tile(const Tile<FloatType> *src_tile, int dst_device_num,
         cudaStream_t stream)
    {
        *this = *src_tile;
        device_num_ = dst_device_num;
        allocate();
        // int retval = omp_target_memcpy(data_, src_tile->data_,
        //                                size(), 0, 0,
        //                                dst_device_num, src_tile->device_num_);
        // assert(retval == 0);
        if (dst_device_num == host_num_) {
            trace_cpu_start();
            cudaError_t error;
            error = cudaSetDevice(src_tile->device_num_);
            assert(error == cudaSuccess);
            error = cudaMemcpyAsync(data_, src_tile->data_, size(),
                                    cudaMemcpyDeviceToHost, stream);
            assert(error == cudaSuccess);
            error = cudaStreamSynchronize(stream);
            assert(error == cudaSuccess);
            trace_cpu_stop("Gray");
        }
        else {
//          trace_cpu_start();
            cudaError_t error;
            error = cudaSetDevice(dst_device_num);
            assert(error == cudaSuccess);
            error = cudaMemcpyAsync(data_, src_tile->data_, size(),
                                    cudaMemcpyHostToDevice, stream);
            assert(error == cudaSuccess);
            error = cudaStreamSynchronize(stream);
            assert(error == cudaSuccess);
//          trace_cpu_stop("LightGray");
        }
    }
    ~Tile() {
        deallocate();
    }

    //---------------------------------------------------------------
    void gemm(blas::Op transa, blas::Op transb, FloatType alpha,
              Tile<FloatType> *a, Tile<FloatType> *b, FloatType beta)
    {
        trace_cpu_start();
        blas::gemm(blas::Layout::ColMajor, transa, transb,
                     mb_, nb_, a->nb_, alpha, a->data_, a->mb_,
                     b->data_, b->mb_, beta, data_, mb_);
        trace_cpu_stop("MediumAquamarine");

        tick(a);
        tick(b);
    }

    //---------------------------------------------------------------
    void potrf(blas::Uplo uplo)
    {
        trace_cpu_start();
        lapack::potrf(blas::Layout::ColMajor, uplo, nb_, data_, nb_);
        trace_cpu_stop("RosyBrown");
    }

    //---------------------------------------------------------------------
    void syrk(blas::Uplo uplo, blas::Op trans,
              FloatType alpha, Tile<FloatType> *a, FloatType beta)
    {
        trace_cpu_start();
        blas::syrk(blas::Layout::ColMajor, uplo, trans,
                   nb_, a->nb_, alpha, a->data_, a->mb_, beta, data_, mb_);
        trace_cpu_stop("CornflowerBlue");

        tick(a);
        tick(a);
    }

    //--------------------------------------------------------------
    void trsm(blas::Side side, blas::Uplo uplo, blas::Op transa,
              blas::Diag diag, FloatType alpha, Tile<FloatType> *a)
    {
        trace_cpu_start();
        blas::trsm(blas::Layout::ColMajor, side, uplo, transa, diag,
                   mb_, nb_, alpha, a->data_, mb_, data_, mb_);
        trace_cpu_stop("MediumPurple");

        tick(a);
    }

    //---------------------------------------------------
    void print()
    {
        for (int64_t m = 0; m < mb_; ++m) {
            for (int64_t n = 0; n < nb_; ++n) {
                printf("%8.2lf", data_[(size_t)mb_*n+m]);
            }
            printf("\n");
        }
        printf("\n");
    }
};

//------------------------------------------------------------------------------
template<typename FloatType>
int Tile<FloatType>::host_num_ = omp_get_initial_device();

template<typename FloatType>
Memory Tile<FloatType>::memory_ = Memory(sizeof(FloatType)*512*512, 1000);

} // namespace slate

#endif // SLATE_TILE_HH
