#include "slate.hh"
#include "slate_BandMatrix.hh"
#include "test.hh"
#include "blas_flops.hh"
#include "lapack_flops.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "print_matrix.hh"

#include "slate_mpi.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#ifdef SLATE_WITH_MKL
extern "C" int MKL_Set_Num_Threads(int nt);
inline int slate_set_num_blas_threads(const int nt) { return MKL_Set_Num_Threads(nt); }
#else
inline int slate_set_num_blas_threads(const int nt) { return -1; }
#endif

//------------------------------------------------------------------------------
template <typename scalar_t> void test_gbtrf_work(Params &params, bool run)
{
    using blas::max;
    using blas::real;
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
    int64_t p = params.p.value();
    int64_t q = params.q.value();
    int64_t nb = params.nb.value();
    int64_t lookahead = params.lookahead.value();
    int64_t panel_threads = params.panel_threads.value();
    lapack::Norm norm = params.norm.value();  // TODO: probably should be specified
    bool check = params.check.value()=='y';
    bool ref = params.ref.value()=='y';
    bool trace = params.trace.value()=='y';
    int verbose = params.verbose.value();
    slate::Target target = char2target(params.target.value());  // TODO: enum

    // mark non-standard output values
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    int64_t Am = m;
    int64_t An = n;

    // Local values
    static int i0=0, i1=1;

    /// // BLACS/MPI variables
    int ictxt, nprow, npcol, myrow, mycol, info;
    /// int descA_tst[9], descA_ref[9];
    int iam=0, nprocs=1;
    /// int iseed = 1;

    // initialize BLACS and ScaLAPACK
    Cblacs_pinfo(&iam, &nprocs);
    assert(p*q <= nprocs);
    Cblacs_get(-1, 0, &ictxt);
    Cblacs_gridinit(&ictxt, "Col", p, q);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    /// // matrix A, figure out local size, allocate, create descriptor, initialize
    /// int64_t mlocA = scalapack_numroc(Am, nb, myrow, i0, nprow);
    /// int64_t nlocA = scalapack_numroc(An, nb, mycol, i0, npcol);
    /// scalapack_descinit(descA_tst, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info);
    /// assert(info==0);
    /// int64_t lldA = (int64_t)descA_tst[8];
    /// std::vector<scalar_t> A_tst(lldA * nlocA);
    /// scalapack_pplrnt(&A_tst[0], Am, An, nb, nb, myrow, mycol, nprow, npcol, mlocA, iseed+1);

    /// // allocate ipiv locally
    /// size_t ipiv_size = (size_t) (lldA + nb);
    /// std::vector<int> ipiv_tst(ipiv_size);

    /// // Create SLATE matrix from the ScaLAPACK layouts
    /// auto A = slate::Matrix<scalar_t>::fromScaLAPACK(Am, An, &A_tst[0], lldA, nb, nprow, npcol, MPI_COMM_WORLD);

    int64_t iseeds[4] = { myrow, mycol, 2, 3 };
    auto A = slate::BandMatrix<scalar_t>(Am, An, kl, ku, nb, p, q, MPI_COMM_WORLD);
    slate::Pivots pivots;

    int64_t klt = slate::ceildiv( kl, nb );
    int64_t kut = slate::ceildiv( ku, nb );
    int64_t jj = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (i >= j - kut && i <= j + klt) {
                A.tileInsert(i, j);
                auto T = A(i, j);
                lapack::larnv(2, iseeds, T.size(), T.data());
                for (int64_t tj = 0; tj < T.nb(); ++tj) {
                    for (int64_t ti = 0; ti < T.mb(); ++ti) {
                        int64_t j_i = (jj + tj) - (ii + ti);
                        if (-kl > j_i || j_i > ku) {
                            T.at( ti, tj ) = 0;
                        }
                    }
                }
            }
            ii += A.tileMb(i);
        }
        jj += A.tileNb(j);
    }

    if (verbose > 1) {
        printf( "rank %d A2 kl %lld, ku %lld\n",
                A.mpiRank(), A.lowerBandwidth(), A.upperBandwidth() );
        print_matrix("A", A);
    }

    /// // if check is required, copy test data and create a descriptor for it
    /// std::vector<scalar_t> A_ref;
    /// std::vector<int> ipiv_ref;
    /// if (check || ref) {
    ///     A_ref.resize(A_tst.size());
    ///     A_ref = A_tst;
    ///     scalapack_descinit(descA_ref, Am, An, nb, nb, i0, i0, ictxt, mlocA, &info);
    ///     assert(info==0);
    ///     ipiv_ref.resize(ipiv_tst.size());
    /// }

    if (trace) slate::trace::Trace::on();
    else slate::trace::Trace::off();

    // run test
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double time = libtest::get_wtime();

    slate::gbtrf(A, pivots, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads}
    });

    MPI_Barrier(MPI_COMM_WORLD);
    {
        slate::trace::Block trace_block("MPI_Barrier");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time_tst = libtest::get_wtime() - time;

    if (trace) slate::trace::Trace::finish();

    // compute and save timing/performance
    ///double gflop = lapack::Gflop<scalar_t>::gbtrf(m, n);
    params.time.value() = time_tst;
    ///params.gflops.value() = gflop / time_tst;

    if (verbose > 1) {
        printf( "rank %d A2 kl %lld, ku %lld\n",
                A.mpiRank(), A.lowerBandwidth(), A.upperBandwidth() );
        print_matrix("A2", A);
    }

    /// if (check || ref) {
    ///     // A comparison with a reference routine from ScaLAPACK
    ///
    ///     // set MKL num threads appropriately for parallel BLAS
    ///     int omp_num_threads;
    ///     #pragma omp parallel
    ///     { omp_num_threads = omp_get_num_threads(); }
    ///     int saved_num_threads = slate_set_num_blas_threads(omp_num_threads);
    ///     int64_t info_ref=0;
    ///
    ///     // allocate work space
    ///     std::vector<real_t> worklange(std::max(mlocA, nlocA));
    ///
    ///     // Run the reference routine
    ///     MPI_Barrier(MPI_COMM_WORLD);
    ///     time = libtest::get_wtime();
    ///     scalapack_pgbtrf(m, n, &A_ref[0], i1, i1, descA_ref, &ipiv_ref[0], &info_ref);
    ///     assert(0 == info_ref);
    ///     MPI_Barrier(MPI_COMM_WORLD);
    ///     double time_ref = libtest::get_wtime() - time;
    ///
    ///     // todo: The IPIV needs to be checked
    ///
    ///     // Norm of the reference result
    ///     real_t A_ref_norm = scalapack_plange(norm2str(norm), Am, An, &A_ref[0], i1, i1, descA_ref, &worklange[0]);
    ///
    ///     // local operation: error = A_ref = A_ref - A_tst;   ipiv_ref = ipiv_ref - ipiv_tst
    ///     blas::axpy(A_ref.size(), -1.0, &A_tst[0], 1, &A_ref[0], 1);
    ///
    ///     // error = norm(error)
    ///     real_t error_norm = scalapack_plange(norm2str(norm), Am, An, &A_ref[0], i1, i1, descA_ref, &worklange[0]);
    ///
    ///     // error = error / reference;
    ///     if (A_ref_norm != 0)
    ///         error_norm /= A_ref_norm;
    ///
    ///     params.ref_time.value() = time_ref;
    ///     params.ref_gflops.value() = gflop / time_ref;
    ///     params.error.value() = error_norm;
    ///
    ///     slate_set_num_blas_threads(saved_num_threads);
    ///
    ///     real_t eps = std::numeric_limits<real_t>::epsilon();
    ///     params.okay.value() = (params.error.value() <= 3*eps);
    /// }

    // Cblacs_exit is commented out because it does not handle re-entering ... some unknown problem
    // Cblacs_exit( 1 ); // 1 means that you can run Cblacs again
}

// -----------------------------------------------------------------------------
void test_gbtrf(Params &params, bool run)
{
    switch(params.datatype.value()) {
    case libtest::DataType::Integer:
        throw std::exception();
        break;

    case libtest::DataType::Single:
        test_gbtrf_work<float> (params, run);
        break;

    case libtest::DataType::Double:
        test_gbtrf_work<double> (params, run);
        break;

    case libtest::DataType::SingleComplex:
        test_gbtrf_work<std::complex<float>> (params, run);
        break;

    case libtest::DataType::DoubleComplex:
        test_gbtrf_work<std::complex<double>> (params, run);
        break;
    }
}
