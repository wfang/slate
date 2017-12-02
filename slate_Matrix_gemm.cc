// TODO: support transa, transb
// #include <omp.h>
#include "slate_Matrix.hh"
#include <sched.h>
#include <vector>

namespace slate {
    
template<typename FloatType>
void Matrix<FloatType>::copyTo_general(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j < nt_; ++j) {
            if (tileIsLocal(i, j))
                (*this)(i, j) =
                    new Tile<FloatType>(tileMb(i), tileNb(j),
                                        &a[(size_t)lda*n+m], lda, memory_);
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

template<typename FloatType>
void Matrix<FloatType>::copyFromFull_general(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j < nt_; ++j) {
            (*this)(i, j)->copyFrom(&a[(size_t)lda*n+m], lda);
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

/* Arguments:
   Ada: (device) Pointer array to temporary buffer for A col tiles
   Bda: (device) Pointer array to temporary buffer for B row tiles

   mm_bcast_gpu will broadcast within row/col and also transfer
   the tile to GPU. 
 */
template<typename FloatType>
void Matrix<FloatType>::mm_bcast_gpu(Matrix &a, Matrix &b, int M, int N,  int k,
				     FloatType* Ad, FloatType* Bd)
{
// #pragma omp task shared(a,b)
    int lmt = 0, lnt = 0;
    cudaError_t error;
    for (int i=0; i<M; i++) {
	if ((i+it_)%p == prow) {
	    if (!a.tileIsLocal(i,k)) {
		// allocate temp tile to receive
		auto *tile = new Tile<FloatType>(a.tileMb(i),a.tileNb(k),a.memory_);
		a(i,k) = tile;
	    }
// #pragma omp task depend(out:adep[i*M + k]) shared(a,b)
	    // #pragma omp task shared(a,b)
	    {
		trace_cpu_start();
		Tile<FloatType>  *tile = a(i,k);
		int count = tile->mb_*tile->nb_;
		// TODO: hardcoded 2d block cyclic distribution
		// TODO: setup life for tile
		int err;
		// #pragma omp critical
		err = MPI_Ibcast(tile->data_, count, MPI_DOUBLE,
				 (jt_+k)%q, mpi_comm_row_, &tile->bcast_request_);
		assert(err == MPI_SUCCESS);
		MPI_Wait(&tile->bcast_request_, MPI_STATUS_IGNORE);
		trace_cpu_stop("Red");
		error = cudaMemcpyAsync(Ad + count*lmt, tile->data_,
					count*sizeof(FloatType),
					cudaMemcpyHostToDevice);
		// assert(error == cudaSuccess);
		if (error != cudaSuccess) {
		    printf("%s:%d error %s\n", __FILE__, __LINE__,
			   cudaGetErrorName(error));
		    printf("R %d lmt %d src %p dest %p\n",
			   mpi_rank_, lmt, tile->data_,
			   Ad + count*lmt);
		    assert(0);
		}
		lmt++;
	    }
	}
    }
    printf("R=%d,k=%d, %d row broadcasted\n", mpi_rank_, k, lmt);
    // broadcast the kth row of b
    // printf("col broadcasting...\n");
// #pragma omp task shared(a,b)
    for (int j=0; j<N; j++) {
	// TODO
	// printf("j=%d\n", j);

	if ((jt_+j)%q == pcol) {
	    // printf("R%d j%d 
	    if (!b.tileIsLocal(k,j)) {
		// printf("creating new tile...\n");
		auto *tile = new Tile<FloatType>(b.tileMb(k), b.tileNb(j),b.memory_);
		b(k,j) = tile;
	    }
// #pragma omp task depend(out:bdep[k*K + j]) shared(a,b)
	    // #pragma omp task shared(a,b)
	    {
		// printf("accessing a(%d,%d)..\n", k, j);
		trace_cpu_start();
		Tile<FloatType> *tile = b(k,j);
		int count = tile->mb_*tile->nb_;
		int err;
		// #pragma omp critical
		err = MPI_Ibcast(tile->data_, count, MPI_DOUBLE,
				 (it_+k)%p, mpi_comm_col_, &tile->bcast_request_);
		assert(err == MPI_SUCCESS);
		MPI_Wait(&tile->bcast_request_, MPI_STATUS_IGNORE);
		trace_cpu_stop("Orange");
		error = cudaMemcpyAsync(Bd+lnt*count, tile->data_,
					count * sizeof(FloatType),
					cudaMemcpyHostToDevice);
		assert(error == cudaSuccess);
		lnt++;
	    }
	}
    }
    printf("R=%d,k=%d, %d col broadcasted\n", mpi_rank_, k, lnt);
    error = cudaDeviceSynchronize();
    assert(error == cudaSuccess);
    // #pragma omp taskwait
}


template<typename FloatType>
void Matrix<FloatType>:: mm_summa_gpu(Matrix &a, Matrix &b, double alpha, double beta, int la)
{
        // the MPI communicator of a, b, and c must be the same.
    if (mpi_comm_ != a.mpi_comm_ || mpi_comm_ != b.mpi_comm_ ||
       mpi_comm_row_ != a.mpi_comm_row_ || mpi_comm_col_ != b.mpi_comm_col_) {
	printf("Matrix::mm: communicator of a,b,c must be the same!\n");
	return;
    }
    // the shape of C=C+A*B must be consistent
    if (mt_ != a.mt_ || nt_ != b.nt_ || a.nt_ != b.mt_) {
	printf("Matrix::mm: shape of a,b,c must be consistent!\n");
	return;
    }
    if (la>0) {
	printf("mm_summa_gpu: no lookahead allowed!\n");
	return;
    }
    int K = a.nt_, M = mt_, N = nt_;
    printf("GPU summa: K=%d,M=%d,N=%d\n", K,M,N);
    printf("GPU summa: R%d (prow,pcol)=(%d,%d)\n", mpi_rank_, prow, pcol);
    char *adep, *bdep, *kdep, depx;
    
    Matrix<FloatType> &c = *this;
    cudaError_t error;
    int lmt = 0, lnt = 0; // number of local rows/columns tiles
    std::vector<int> lm2gm, ln2gn; // lm2gm[li]=global index i
    for (int i=0; i<M; i++) 
	if (i%p == prow) {
	    lmt++;
	    lm2gm.push_back(i);
	}

    for (int j=0; j<N; j++)
	if (j%q == pcol) {
	    lnt++;
	    ln2gn.push_back(j);
	}
    
    printf("R=%d, lmt=%d, lnt=%d\n", mpi_rank_, lmt, lnt);
    int tile_size = tileNb(0)*tileNb(0);
    FloatType *Ad, *Bd, *Cd; // device memory for matrix data
    error = cudaMalloc((void**) &Cd,  (lmt * lnt) * tile_size * sizeof(FloatType));
    assert(error == cudaSuccess);
    error = cudaMalloc((void**) &Ad,  (lmt * 1) * tile_size * sizeof(FloatType));
    assert(error == cudaSuccess);
    error = cudaMalloc((void**) &Bd,  (1 * lnt) * tile_size * sizeof(FloatType));
    assert(error == cudaSuccess);

    // First move all of local C to the GPU
    for (int li=0; li<lmt; li++) {
	int i = lm2gm[li];
	for (int lj=0; lj<lnt; lj++) {
	    int j = ln2gn[lj];
	    // printf("li %d lj %d i %d j %d\n", li, lj, i, j);
	    assert(tileIsLocal(i,j));
	    error = cudaMemcpyAsync(Cd + tile_size * (li+lj*lmt), c(i,j)->data_,
				    tile_size * sizeof(FloatType),
				    cudaMemcpyHostToDevice);
	    assert(error == cudaSuccess);
	}
    }
    // TODO: Is this sync really needed?
    cudaDeviceSynchronize(); // wait for all data transfer ends.

    //prepare the pointer to tiles in Acol, Brow, and C;
    const FloatType **Ad_array, **Bd_array;
    FloatType **Cd_array;
    FloatType **Ah_array = (FloatType**)malloc( lmt*lnt*sizeof(FloatType*) );
    cudaMalloc(&Ad_array, lmt*lnt*sizeof(FloatType*));
    FloatType **Bh_array = (FloatType**)malloc( lmt*lnt*sizeof(FloatType*) );
    cudaMalloc(&Bd_array, lmt*lnt*sizeof(FloatType*));
    FloatType **Ch_array = (FloatType**)malloc( lmt*lnt*sizeof(FloatType*) );
    cudaMalloc(&Cd_array, lmt*lnt*sizeof(FloatType*));
    // row major
    for (int li=0; li<lmt; li++) {
	for (int lj=0; lj<lnt; lj++) {
	    Ch_array[li+lj*lmt] = &Cd[tile_size * (li+lj*lmt)];
	    Ah_array[li+lj*lmt] = &Ad[tile_size * li];
	    Bh_array[li+lj*lmt] = &Bd[tile_size * lj];
	}
    }
    error = cudaMemcpyAsync(Cd_array, Ch_array,
			    sizeof(FloatType*) * lmt * lnt,
			    cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
    error = cudaMemcpyAsync(Ad_array, Ah_array,
			    sizeof(FloatType*) * lmt * lnt,
			    cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
    error = cudaMemcpyAsync(Bd_array, Bh_array,
			    sizeof(FloatType*) * lmt * lnt,
			    cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
	printf ("CUBLAS initialization failed\n");
	assert(0);
    }


    #pragma omp parallel
    #pragma omp master
    {
	for (int i=0; i<la; i++) {
            // #pragma omp task shared(a,b) \
	    // 	depend(out:kdep[i]) depend(inout:depx) final(1)
	    mm_bcast_gpu(a,b,M,N,i,Ad,Bd);
			 // const_cast<FloatType**>(Ad_array),
			 // const_cast<FloatType**>(Bd_array));
	}

	for (int k=0; k<K; k++) {
	    if (k+la < K) {
                // #pragma omp task shared(a,b) \
		//     depend(out:kdep[k+la]) depend(inout:depx) final(1)
		mm_bcast_gpu(a,b,M,N,k+la, Ad, Bd);
			     // const_cast<FloatType**>(Ad_array),
			     // const_cast<FloatType**>(Bd_array));
	    }
	    // At this point, the k-th column of A and k-th row of B
	    // should already arrive at GPU. Prepare to launch batch gemm
	    // on GPU.
	    // TODO: extend to multiple devices
	    int device = 0; // one device for now.

	    error = cudaSetDevice(device);
	    assert(error == cudaSuccess);
	    int batchCount = lmt*lnt;
	    // Constructing the A,B,C pointer arrays.
	    // Actually, not needed because the static allocation in Ad/Bd/Cd
	    int nb = tileNb(0);
	    FloatType fone = 1;
	    if (k == 0) {
		cublasStatus_t status =
		    cublasDgemmBatched(
			// cublas_handle_[device],
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			nb, nb, nb,
			&alpha, Ad_array, nb,
			Bd_array, nb,
			&beta, Cd_array, nb,
			lmt*lnt);
		assert(status == CUBLAS_STATUS_SUCCESS);
	    } else {
		cublasStatus_t status =
		    cublasDgemmBatched(
			handle,
			// cublas_handle_[device],
			CUBLAS_OP_N, CUBLAS_OP_N,
			nb, nb, nb,
			&alpha, Ad_array, nb,
			Bd_array, nb,
			&fone, Cd_array, nb,
			lmt*lnt);
		assert(status == CUBLAS_STATUS_SUCCESS);
	    }

	    error = cudaDeviceSynchronize();
	    assert(error == cudaSuccess);
	}
		
    }
    cudaDeviceSynchronize();
    // Copy C back to host
    int nb = tileNb(0);
    for (int li=0; li<lmt; li++) {
	int i = lm2gm[li];
	for (int lj=0; lj<lnt; lj++) {
	    int j = ln2gn[lj];
	    assert(tileIsLocal(i,j));
	    cudaMemcpyAsync(c(i,j)->data_, Cd+ (li+lj*lmt)*nb*nb,
			    nb*nb*sizeof(FloatType),
			    cudaMemcpyDeviceToHost);
	}
    }
    cudaDeviceSynchronize();

    free(Ah_array); cudaFree(Ad_array);
    free(Bh_array); cudaFree(Bd_array);
    free(Ch_array); cudaFree(Cd_array);
}

template<typename FloatType>
void Matrix<FloatType>:: mm_bcast(Matrix &a, Matrix &b, int M, int N,  int k)
{
// #pragma omp task shared(a,b)
    for (int i=0; i<M; i++) {
	if ((i+it_)%p == prow) {
	    if (!a.tileIsLocal(i,k)) {
		// allocate temp tile to receive
		auto *tile = new Tile<FloatType>(a.tileMb(i),a.tileNb(k),a.memory_);
		a(i,k) = tile;
	    }
// #pragma omp task depend(out:adep[i*M + k]) shared(a,b)
	    // #pragma omp task shared(a,b)
	    {
		trace_cpu_start();
		Tile<FloatType>  *tile = a(i,k);
		int count = tile->mb_*tile->nb_;
		// TODO: hardcoded 2d block cyclic distribution
		// TODO: setup life for tile
		int err;
		// #pragma omp critical
		err = MPI_Ibcast(tile->data_, count, MPI_DOUBLE,
				 (jt_+k)%q, mpi_comm_row_, &tile->bcast_request_);
		assert(err == MPI_SUCCESS);
		MPI_Wait(&tile->bcast_request_, MPI_STATUS_IGNORE);
		trace_cpu_stop("Red");
	    }
	}
    }
    // broadcast the kth row of b
    // printf("col broadcasting...\n");
// #pragma omp task shared(a,b)
    for (int j=0; j<N; j++) {
	// TODO
	// printf("j=%d\n", j);

	if ((jt_+j)%q == pcol) {
	    // printf("R%d j%d 
	    if (!b.tileIsLocal(k,j)) {
		// printf("creating new tile...\n");
		auto *tile = new Tile<FloatType>(b.tileMb(k), b.tileNb(j),b.memory_);
		b(k,j) = tile;
	    }
// #pragma omp task depend(out:bdep[k*K + j]) shared(a,b)
	    // #pragma omp task shared(a,b)
	    {
		// printf("accessing a(%d,%d)..\n", k, j);
		trace_cpu_start();
		Tile<FloatType> *tile = b(k,j);
		int count = tile->mb_*tile->nb_;
		int err;
		// #pragma omp critical
		err = MPI_Ibcast(tile->data_, count, MPI_DOUBLE,
				 (it_+k)%p, mpi_comm_col_, &tile->bcast_request_);
		assert(err == MPI_SUCCESS);
		MPI_Wait(&tile->bcast_request_, MPI_STATUS_IGNORE);
		trace_cpu_stop("Orange");
	    }
	}
    }
    // #pragma omp taskwait
}

    
template<typename FloatType>
void Matrix<FloatType>::mm_summa_blocking_pipelined(Matrix &a, Matrix &b, double alpha, double beta, int la)
{
    
    // the MPI communicator of a, b, and c must be the same.
    if (mpi_comm_ != a.mpi_comm_ || mpi_comm_ != b.mpi_comm_ ||
       mpi_comm_row_ != a.mpi_comm_row_ || mpi_comm_col_ != b.mpi_comm_col_) {
	printf("Matrix::mm: communicator of a,b,c must be the same!\n");
	return;
    }
    // the shape of C=C+A*B must be consistent
    if (mt_ != a.mt_ || nt_ != b.nt_ || a.nt_ != b.mt_) {
	printf("Matrix::mm: shape of a,b,c must be consistent!\n");
	return;
    }
    int K = a.nt_, M = mt_, N = nt_;
    printf("summa: K=%d,M=%d,N=%d\n", K,M,N);
    printf("summa: R%d (prow,pcol)=(%d,%d)\n", mpi_rank_, prow, pcol);
    char *adep, *bdep, *kdep, depx;


#pragma omp parallel
#pragma omp master
// #pragma task untied
    {
	// pipeline head
    for (int i=0; i<la; i++) {
#pragma omp task shared(a,b) depend(out:kdep[i])  depend(inout: depx) final(1)
	mm_bcast(a,b,M,N,i);
    }

    for (int k=0; k<K; k++) { // phase k
	// The order of the communication tasks must be enforced. Either use a dependence
	// clause or if(0) to force execution before continuing.
	// On summitdev, if(0) seems to perform better.
	// The scheduler is not particularly smart on summitdev as it schedules too much
	// computation tasks on the "communication task", thus making a part of the comm
	// part of the critical path.

	if (k+la<K) {
#pragma omp task shared(a,b) depend(out:kdep[k+la])  depend(inout: depx) final(1)
	    mm_bcast(a, b, M, N, k+la);
	}
	// this taskwait is very important on saturn, 256*80@4nodes. 
	#pragma omp taskwait
	// do the trailing matrix update
	for (int i=0; i<M; i++) {
	    for (int j=0; j<N; j++) {
                // #pragma omp task depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
		// #pragma omp task
		{
		    if (tileIsLocal(i,j)) {
			// The first iteration does C = alpha*A1*B1 + beta*C;
			// The rest does C = alpha*Ak*Bk + C
                        // #pragma omp task shared(a,b) depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
			// #pragma omp task
// #pragma omp task shared(a,b) depend(in:adep[k])
#pragma omp task shared(a,b) depend(in:kdep[k])
			{
			    // int cpu = sched_getcpu();
			    // printf("Updating C(%d,%d) on rank %d cpu# %d\n", i, j, mpi_rank_, cpu);
			    omp_set_lock(&(*this)(i,j)->access_lck);
			    if (k==0)
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), beta);
			    else 
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), 1.0);
			    omp_unset_lock(&(*this)(i,j)->access_lck);
			}
		    }
		}
	    }
	}

    } // end of loop k.
    } // end of OpenMP parallel region. Implicit synchronization here.
}
template<typename FloatType>
void Matrix<FloatType>::mm_summa(Matrix &a, Matrix &b, double alpha, double beta)
{
    
    // the MPI communicator of a, b, and c must be the same.
    if (mpi_comm_ != a.mpi_comm_ || mpi_comm_ != b.mpi_comm_ ||
       mpi_comm_row_ != a.mpi_comm_row_ || mpi_comm_col_ != b.mpi_comm_col_) {
	printf("Matrix::mm: communicator of a,b,c must be the same!\n");
	return;
    }
    // the shape of C=C+A*B must be consistent
    if (mt_ != a.mt_ || nt_ != b.nt_ || a.nt_ != b.mt_) {
	printf("Matrix::mm: shape of a,b,c must be consistent!\n");
	return;
    }
    int K = a.nt_, M = mt_, N = nt_;
    printf("summa: K=%d,M=%d,N=%d\n", K,M,N);
    printf("summa: R%d (prow,pcol)=(%d,%d)\n", mpi_rank_, prow, pcol);
    char *adep, *bdep;
    // TODO: implement pipeline to overlap communication with computation
    #pragma omp parallel
    #pragma omp master
    for (int k=0; k<K; k++) { // phase k
	// broadcast the kth col ,of a
	printf("iteration k=%d\n",k);
	// printf("row broadcasting...\n");
	for (int i=0; i<M; i++) {
	    if ((i+it_)%p == prow) {
		if (!a.tileIsLocal(i,k)) {
		    // allocate temp tile to receive
		    auto *tile = new Tile<FloatType>(a.tileMb(i),a.tileNb(k),a.memory_);
		    a(i,k) = tile;
		}
                // #pragma omp task depend(out:adep[i*M + k]) shared(a,b)
                // #pragma omp task shared(a,b)
		{
		    trace_cpu_start();
		    Tile<FloatType>  *tile = a(i,k);
		    int count = tile->mb_*tile->nb_;
		    // TODO: hardcoded 2d block cyclic distribution
		    // TODO: setup life for tile
		    int err;
		    // printf("R%d: row bcast (i,k)=(%d,%d) root=%d\n", mpi_rank_, i, k,(jt_+k)%q);
                    // #pragma omp critical
		    err = MPI_Bcast(tile->data_, count, MPI_DOUBLE,
				    (jt_+k)%q, mpi_comm_row_);
		    assert(err == MPI_SUCCESS);
		    trace_cpu_stop("Red");
		}
	    }
	}
	// broadcast the kth row of b
	for (int j=0; j<N; j++) {
	    // TODO

	    if ((jt_+j)%q == pcol) {
		if (!b.tileIsLocal(k,j)) {
		    auto *tile = new Tile<FloatType>(b.tileMb(k), b.tileNb(j),b.memory_);
		    b(k,j) = tile;
		}
                // #pragma omp task depend(out:bdep[k*K + j]) shared(a,b)
                // #pragma omp task shared(a,b)
		{
		    trace_cpu_start();
		    Tile<FloatType> *tile = b(k,j);
		    int count = tile->mb_*tile->nb_;
		    int err;
                    // #pragma omp critical
		    err = MPI_Bcast(tile->data_, count, MPI_DOUBLE,
				    (it_+k)%p, mpi_comm_col_);
		    assert(err == MPI_SUCCESS);
		    trace_cpu_stop("Orange");
		}
	    }
	}
	#pragma omp taskwait
	// do the trailing matrix update
	// printf("update matrix C...\n");
	for (int i=0; i<M; i++) {
	    for (int j=0; j<N; j++) {
                // #pragma omp task depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
		// #pragma omp task
		{
		    if (tileIsLocal(i,j)) {
			// The first iteration does C = alpha*A1*B1 + beta*C;
			// The rest does C = alpha*Ak*Bk + C
                        // #pragma omp task shared(a,b) depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
			#pragma omp task
			{
			    int cpu = sched_getcpu();
			    // printf("Updating C(%d,%d) on rank %d cpu# %d\n", i, j, mpi_rank_, cpu);
			    if (k==0)
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), beta);
			    else 
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), 1.0);
			}
		    }
		}
	    }
	}
	// #pragma omp taskwait
    } // end of OpenMP parallel region. Implicit synchronization here. 
}

// non-blocking collective SUMMA
template<typename FloatType>
void Matrix<FloatType>::mm_summa_nb(Matrix &a, Matrix &b, double alpha, double beta)
{
    
    // the MPI communicator of a, b, and c must be the same.
    if (mpi_comm_ != a.mpi_comm_ || mpi_comm_ != b.mpi_comm_ ||
       mpi_comm_row_ != a.mpi_comm_row_ || mpi_comm_col_ != b.mpi_comm_col_) {
	printf("Matrix::mm: communicator of a,b,c must be the same!\n");
	return;
    }
    // the shape of C=C+A*B must be consistent
    if (mt_ != a.mt_ || nt_ != b.nt_ || a.nt_ != b.mt_) {
	printf("Matrix::mm: shape of a,b,c must be consistent!\n");
	return;
    }
    int K = a.nt_, M = mt_, N = nt_;
    printf("summa: K=%d,M=%d,N=%d\n", K,M,N);
    char *adep, *bdep;
    // TODO: implement pipeline to overlap communication with computation
    #pragma omp parallel
    #pragma omp master
    for (int k=0; k<K; k++) { // phase k
	// broadcast the kth col ,of a
	// printf("iteration k=%d\n",k);
	// printf("row broadcasting...\n");
	mm_bcast(a, b, M, N, k);
	#pragma omp taskwait
	// do the trailing matrix update
	// printf("update matrix C...\n");
	for (int i=0; i<M; i++) {
	    for (int j=0; j<N; j++) {
                // #pragma omp task depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
		// #pragma omp task
		{
		    // printf("a(%d,%d)->mb_=%d, data_=%p, b(%d,%d)->nb_=%d, data=%p\n",
		    // 	   i, k, a(i,k)->mb_, a(i,k)->data_,
		    // 	   k, j, b(k,j)->nb_, b(k,j)->data_);
		    // for (auto it=a.tiles_->begin(); it!=a.tiles_->end(); it++) {
		    // 	printf("a(%d,%d) ", std::get<0>(it->first),
		    // 	       std::get<1>(it->first));
		    // }
		    // printf("\n");
		    if (tileIsLocal(i,j)) {

			// printf("rank %d count a(%d,%d)=%d b(%d,%d)=%d c(%d,%d)=%d \n",
			//        mpi_rank_,
			//        i,k,a.tiles_->count({i,k,host_num_}),
			//        k,j,b.tiles_->count({k,j,host_num_}),
			//        i,j,tiles_->count({i,j,host_num_}));
			// The first iteration does C = alpha*A1*B1 + beta*C;
			// The rest does C = alpha*Ak*Bk + C
// #pragma omp task shared(a,b) depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
                        #pragma omp task shared(a,b)
			{
			    // int cpu = sched_getcpu();
			    // printf("Updating C(%d,%d) on rank %d cpu# %d\n", i, j, mpi_rank_, cpu);
			    trace_cpu_start();
			    omp_set_lock(&a(i,k)->bcast_req_lck);
			    // #pragma omp critical
			    MPI_Wait(&a(i,k)->bcast_request_, MPI_STATUS_IGNORE);
			    omp_unset_lock(&a(i,k)->bcast_req_lck);
			    trace_cpu_stop("Red");
			    
			    trace_cpu_start();
			    omp_set_lock(&b(k,j)->bcast_req_lck);
			    // #pragma omp critical
			    MPI_Wait(&b(k,j)->bcast_request_, MPI_STATUS_IGNORE);
			    omp_unset_lock(&b(k,j)->bcast_req_lck);
			    trace_cpu_stop("Orange");

			    omp_set_lock(&(*this)(i,j)->access_lck);
			    if (k==0)
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), beta);
			    else 
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), 1.0);
			    omp_unset_lock(&(*this)(i,j)->access_lck);
			}
		    }
		}
	    }
	}
	// #pragma omp taskwait
    } // end of parallel region
}
  // auxillary method for bcast
template<typename FloatType>
void Matrix<FloatType>:: mm_Ibcast(Matrix &a, Matrix &b, int M, int N,  int k)
{
    
    for (int i=0; i<M; i++) {
	if ((i+it_)%p == prow) {
	    if (!a.tileIsLocal(i,k)) {
		// allocate temp tile to receive
		auto *tile = new Tile<FloatType>(a.tileMb(i),a.tileNb(k),a.memory_);
		a(i,k) = tile;
	    }
// #pragma omp task depend(out:adep[i*M + k]) shared(a,b)
	    // #pragma omp task shared(a,b)
	    {
		trace_cpu_start();
		Tile<FloatType>  *tile = a(i,k);
		int count = tile->mb_*tile->nb_;
		// TODO: hardcoded 2d block cyclic distribution
		// TODO: setup life for tile
		int err;
		// #pragma omp critical
		err = MPI_Ibcast(tile->data_, count, MPI_DOUBLE,
				 (jt_+k)%q, mpi_comm_row_, &tile->bcast_request_);
		assert(err == MPI_SUCCESS);
		// MPI_Wait(&tile->bcast_request_, MPI_STATUS_IGNORE);
		trace_cpu_stop("Red");
	    }
	}
    }
    // broadcast the kth row of b
    // printf("col broadcasting...\n");
    for (int j=0; j<N; j++) {
	// TODO
	// printf("j=%d\n", j);

	if ((jt_+j)%q == pcol) {
	    // printf("R%d j%d 
	    if (!b.tileIsLocal(k,j)) {
		// printf("creating new tile...\n");
		auto *tile = new Tile<FloatType>(b.tileMb(k), b.tileNb(j),b.memory_);
		b(k,j) = tile;
	    }
// #pragma omp task depend(out:bdep[k*K + j]) shared(a,b)
	    // #pragma omp task shared(a,b)
	    {
		// printf("accessing a(%d,%d)..\n", k, j);
		trace_cpu_start();
		Tile<FloatType> *tile = b(k,j);
		int count = tile->mb_*tile->nb_;
		int err;
		// #pragma omp critical
		err = MPI_Ibcast(tile->data_, count, MPI_DOUBLE,
				 (it_+k)%p, mpi_comm_col_, &tile->bcast_request_);
		assert(err == MPI_SUCCESS);
		// MPI_Wait(&tile->bcast_request_, MPI_STATUS_IGNORE);
		trace_cpu_stop("Orange");
	    }
	}
    }

}
// non-blocking collective SUMMA; pipelined
template<typename FloatType>
void Matrix<FloatType>::mm_summa_pl(Matrix &a, Matrix &b, double alpha, double beta, int la)
{
    
    // the MPI communicator of a, b, and c must be the same.
    if (mpi_comm_ != a.mpi_comm_ || mpi_comm_ != b.mpi_comm_ ||
       mpi_comm_row_ != a.mpi_comm_row_ || mpi_comm_col_ != b.mpi_comm_col_) {
	printf("Matrix::mm: communicator of a,b,c must be the same!\n");
	return;
    }
    // the shape of C=C+A*B must be consistent
    if (mt_ != a.mt_ || nt_ != b.nt_ || a.nt_ != b.mt_) {
	printf("Matrix::mm: shape of a,b,c must be consistent!\n");
	return;
    }
    int K = a.nt_, M = mt_, N = nt_;
    printf("summa: K=%d,M=%d,N=%d, lookahead=%d\n", K,M,N,la);
    char *adep, *bdep;
    // TODO: implement pipeline to overlap communication with computation
    // #pragma omp parallel
    #pragma omp master
    {
	for (int i=0; i<la; i++)
	    mm_bcast(a,b,M,N,i);
    }
    #pragma omp parallel
    #pragma omp master
    for (int k=0; k<K; k++) { // phase k
	if (k+la<K) mm_Ibcast(a, b, M, N, k+la);
	// broadcast the kth col ,of a
	// printf("iteration k=%d\n",k);
	// printf("row broadcasting...\n");

	#pragma omp taskwait
	// do the trailing matrix update
	// printf("update matrix C...\n");
	for (int i=0; i<M; i++) {
	    for (int j=0; j<N; j++) {
                // #pragma omp task depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
		// #pragma omp task
		{
		    // printf("a(%d,%d)->mb_=%d, data_=%p, b(%d,%d)->nb_=%d, data=%p\n",
		    // 	   i, k, a(i,k)->mb_, a(i,k)->data_,
		    // 	   k, j, b(k,j)->nb_, b(k,j)->data_);
		    // for (auto it=a.tiles_->begin(); it!=a.tiles_->end(); it++) {
		    // 	printf("a(%d,%d) ", std::get<0>(it->first),
		    // 	       std::get<1>(it->first));
		    // }
		    // printf("\n");
		    if (tileIsLocal(i,j)) {

			// printf("rank %d count a(%d,%d)=%d b(%d,%d)=%d c(%d,%d)=%d \n",
			//        mpi_rank_,
			//        i,k,a.tiles_->count({i,k,host_num_}),
			//        k,j,b.tiles_->count({k,j,host_num_}),
			//        i,j,tiles_->count({i,j,host_num_}));
			// The first iteration does C = alpha*A1*B1 + beta*C;
			// The rest does C = alpha*Ak*Bk + C
// #pragma omp task shared(a,b) depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
                        #pragma omp task shared(a,b)
			{
			    // int cpu = sched_getcpu();
			    // printf("Updating C(%d,%d) on rank %d cpu# %d\n", i, j, mpi_rank_, cpu);
			    trace_cpu_start();
			    omp_set_lock(&a(i,k)->bcast_req_lck);
			    // #pragma omp critical
			    MPI_Wait(&a(i,k)->bcast_request_, MPI_STATUS_IGNORE);
			    omp_unset_lock(&a(i,k)->bcast_req_lck);
			    trace_cpu_stop("Red");
			    
			    trace_cpu_start();
			    omp_set_lock(&b(k,j)->bcast_req_lck);
			    // #pragma omp critical
			    MPI_Wait(&b(k,j)->bcast_request_, MPI_STATUS_IGNORE);
			    omp_unset_lock(&b(k,j)->bcast_req_lck);
			    trace_cpu_stop("Orange");

			    omp_set_lock(&(*this)(i,j)->access_lck);
			    if (k==0)
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), beta);
			    else 
				(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
						   a(i,k), b(k,j), 1.0);
			    omp_unset_lock(&(*this)(i,j)->access_lck);
			}
		    }
		}
	    }
	}
	// #pragma omp taskwait
    } // end of parallel region
}



template
void Matrix<double>::copyFromFull_general(double *a, int64_t lda);
template
void Matrix<double>::copyTo_general(double *a, int64_t lda);
template
void Matrix<double>::mm_summa(Matrix &a, Matrix &b, double alpha, double beta);
template
void Matrix<double>::mm_summa_nb(Matrix &a, Matrix &b, double alpha, double beta);
template
void Matrix<double>::mm_summa_pl(Matrix &a, Matrix &b, double alpha, double beta, int la);
template
void Matrix<double>::mm_summa_blocking_pipelined(Matrix &a, Matrix &b, double alpha, double beta, int la);
template
void Matrix<double>::mm_summa_gpu(Matrix &a, Matrix &b, double alpha, double beta, int la);
}
