
#include "slate_Matrix.hh"
#include "slate_mm.hh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#include <omp.h>

extern "C" void trace_on();
extern "C" void trace_off();
extern "C" void trace_finish();
void print_lapack_matrix(int64_t m, int64_t n, double *a, int64_t lda,
                         int64_t mb, int64_t nb);
void diff_lapack_matrices(int64_t m, int64_t n, double *a, int64_t lda,
                          double *b, int64_t ldb,
                          int64_t mb, int64_t nb);

//------------------------------------------------------------------------------
int main (int argc, char *argv[])
{
    assert(argc == 7);
    int64_t nb = atoll(argv[1]);
    int64_t nt = atoll(argv[2]);
    int64_t p = atoll(argv[3]);
    int64_t q = atoll(argv[4]);
    int64_t lookahead = atoll(argv[5]);
    int64_t test = atoll(argv[6]);
    int64_t n = nb*nt;
    int64_t lda = n;

    //------------------------------------------------------
    int mpi_rank = 0;
    int mpi_size = 1;
    int provided;
    int retval;
//  assert(MPI_Init(&argc, &argv) == MPI_SUCCESS);
    retval = MPI_Init_thread(nullptr, nullptr,
                             MPI_THREAD_MULTIPLE, &provided);
    assert(retval == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_MULTIPLE);

    assert(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS);
    assert(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) == MPI_SUCCESS);
    assert(mpi_size == p*q);

    MPI_Comm row_comm, col_comm;
    int prow, pcol;
    prow = mpi_rank % p;
    pcol = mpi_rank / p;
    printf("Rank %d, prow %d, pcol %d\n", mpi_rank, prow, pcol);
    int err;
    err = MPI_Comm_split(MPI_COMM_WORLD, prow, pcol, &row_comm);
    assert(err==MPI_SUCCESS);
    err = MPI_Comm_split(MPI_COMM_WORLD, pcol, prow, &col_comm);
    assert(err==MPI_SUCCESS);

    double *a1 = nullptr;
    double *a2 = nullptr;
    double *b2 = nullptr;
    double *c2 = nullptr;
    double *c1 = nullptr;
    if (test) {
        int seed[] = {0, 0, 0, 1};
        a1 = new double[nb*nb*nt*nt];
        lapack::larnv(1, seed, lda*n, a1);
	c1 = new double[nb*nb*nt*nt];
        // for (int64_t i = 0; i < n; ++i)
        //     a1[i*lda+i] += sqrt(n);

        if (mpi_rank == 0) {
            a2 = new double[nb*nb*nt*nt];
	    b2 = new double[nb*nb*nt*nt];
	    c2 = new double[nb*nb*nt*nt];
            memcpy(a2, a1, sizeof(double)*lda*n);
	    memcpy(b2, a1, sizeof(double)*lda*n);
	    memcpy(c2, a1, sizeof(double)*lda*n);
        }
    }

    //------------------------------------------------------
    trace_off();
    // slate::Matrix<double> temp(n, n, a1, lda, nb, nb, MPI_COMM_WORLD, p, q);
    // temp.potrf(blas::Uplo::Lower);

//  slate::Matrix<double> a(n, n, a1, lda, nb, nb, MPI_COMM_WORLD, p, q);
    printf("creating matrix\n");
    slate::Matrix<double> a(n, n, a1, lda, nb, nb, MPI_COMM_WORLD, row_comm, col_comm, p, q);
    slate::Matrix<double> b(n, n, a1, lda, nb, nb, MPI_COMM_WORLD, row_comm, col_comm, p, q);
    slate::Matrix<double> c(n, n, a1, lda, nb, nb, MPI_COMM_WORLD, row_comm, col_comm, p, q);
    printf("matrix created\n");
    trace_on();

    trace_cpu_start();
    MPI_Barrier(MPI_COMM_WORLD);
    trace_cpu_stop("Black");

    double start = omp_get_wtime();
    // a.potrf(blas::Uplo::Lower, lookahead);
    double alpha = 1.0, beta = 0.0;

    trace_cpu_start();
    MPI_Barrier(MPI_COMM_WORLD);
    trace_cpu_stop("Black");
    c.mm_summa(a,b,alpha, beta);
    double time = omp_get_wtime()-start;
    trace_finish();

    if (test) {
	c.gather_general();
    }

    //------------------------------------------------------
    if (mpi_rank == 0) {

        double gflops = (double)nb*nb*nb*nt*nt*nt/3.0/time/1000000000.0;
        printf("\t%.0lf GFLOPS\n", gflops);
        fflush(stdout);

        // retval = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a2, lda);
        // assert(retval == 0);
	blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
		   n, n, n, alpha, a2, lda, b2, lda, beta, c2, lda);


        // a.copyFromFull(a1, lda);
	c.copyFromFull(c1, lda);
        // diff_lapack_matrices(n, n, a1, lda, a2, lda, nb, nb);
	diff_lapack_matrices(n, n, c1, lda, c2, lda, nb, nb);

        // cblas_daxpy((size_t)lda*n, -1.0, a1, 1, a2, 1);
        // double norm = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, a1, lda);
        // double error = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, a2, lda);
	double error = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, c2, lda);
        // delete[] a2;

        // if (norm != 0)
        //     error /= norm;
        // printf("\t%le\n", error);
    }

    //------------------------------------------------------
//  delete[] a1;
    return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
void print_lapack_matrix(int64_t m, int64_t n, double *a, int64_t lda,
                         int64_t mb, int64_t nb)
{
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            printf("%8.2lf", a[(size_t)lda*j+i]);
            if ((j+1)%nb == 0)
                printf(" |");
        }
        printf("\n");
        if ((i+1)%mb == 0) {
            for (int64_t j = 0; j < (n+1)*8; ++j) {
                printf("-");
            }
            printf("\n");        
        }
    }
    printf("\n");
}

//------------------------------------------------------------------------------
void diff_lapack_matrices(int64_t m, int64_t n, double *a, int64_t lda,
                          double *b, int64_t ldb, int64_t mb, int64_t nb)
{
    for (int64_t i = 0; i < m; ++i) {
        if (i%mb == 2)
            i += mb-4;
        for (int64_t j = 0; j < n; ++j) {
            if (j%nb == 2)
                j += nb-4;
            double error = a[(size_t)lda*j+i] - b[(size_t)lda*j+i];
            printf("%c", error < 0.000001 ? '.' : '#');
            if ((j+1)%nb == 0)
                printf("|");
        }
        printf("\n");
        if ((i+1)%mb == 0) {
            for (int64_t j = 0; j < (n/nb)*5; ++j) {
                printf("-");
            }
            printf("\n");        
        }
    }
    printf("\n");
}
