// TODO: support transa, transb
namespace slate {
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
    char *adep, *bdep;
    // TODO: implement pipeline to overlap communication with computation
    for (int k=0; k<K; k++) { // phase k
	// broadcast the kth col ,of a
	printf("iteration k=%d\n",k);
	printf("row broadcasting...\n");
	for (int i=0; i<M; i++) {
	    if ((i+it_)%p == prow) {
		if (!a.tileIsLocal(i,k)) {
		    // allocate temp tile to receive
		    auto *tile = new Tile<FloatType>(a.tileMb(i),a.tileNb(k));
		    a(i,k) = tile;
		}
		// #pragma omp task depend(out:adep[i*M + k])
		{
		    auto tile = a(i,k);
		    int count = tile->mb_*tile->nb_;
		    // TODO: hardcoded 2d block cyclic distribution
		    // TODO: setup life for tile
		    int err;
                    #pragma omp critical
		    err = MPI_Bcast(tile->data_, count, MPI_DOUBLE,
				    (jt_+k)%q, mpi_comm_row_);
		    assert(err == MPI_SUCCESS);
		}
	    }
	}
	// broadcast the kth row of b
	printf("col broadcasting...\n");
	for (int j=0; j<N; j++) {
	    // TODO
	    // printf("j=%d\n", j);

	    if ((jt_+j)%q == pcol) {
		// printf("R%d j%d 
		if (!b.tileIsLocal(k,j)) {
		    // printf("creating new tile...\n");
		    auto *tile = new Tile<FloatType>(b.tileMb(k), b.tileNb(j));
		    b(k,j) = tile;
		}
		// #pragma omp task depend(out:bdep[k*K + j])
		{
		    // printf("accessing a(%d,%d)..\n", k, j);
		    auto *tile = b(k,j);
		    int count = tile->mb_*tile->nb_;
		    int err;
                    #pragma omp critical
		    err = MPI_Bcast(tile->data_, count, MPI_DOUBLE,
				    (it_+k)%p, mpi_comm_col_);
		    assert(err == MPI_SUCCESS);
		}
	    }
	}
	// do the trailing matrix update
	printf("update matrix C...\n");
	for (int i=0; i<M; i++) {
	    for (int j=0; j<N; j++) {
                // #pragma omp task depend(in:adep[i*M+k]) depend(in:bdep[k*K+j])
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
			(*this)(i,j)->gemm(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
					   a(i,k), b(k,j), beta);
		    }
		}
	    }
	}
    }
}

}
