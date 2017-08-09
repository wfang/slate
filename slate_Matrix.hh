
#ifndef SLATE_MATRIX_HH
#define SLATE_MATRIX_HH

#include "slate_Tile.hh"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <mpi.h>
#include <omp.h>

namespace slate {

//------------------------------------------------------------------------------
template<typename FloatType>
class Matrix {
public:
    int64_t it_; ///< first row of tiles
    int64_t jt_; ///< first column of tiles
    int64_t mt_; ///< number of tile rows
    int64_t nt_; ///< number of tile columns

    // TODO: replace by unordered_map
    std::map<std::pair<int64_t, int64_t>, Tile<FloatType>*> *tiles_;

    Matrix(int64_t m, int64_t n, double *a, int64_t lda,
           int64_t mb, int64_t nb);

    Matrix(int64_t m, int64_t n, double *a, int64_t lda,
           int64_t mb, int64_t nb, MPI_Comm mpi_comm, int64_t p, int64_t q);

    Matrix(const Matrix &a, int64_t it, int64_t jt, int64_t mt, int64_t nt);

    void copyTo(FloatType *a, int64_t lda);
    void copyFrom(FloatType *a, int64_t lda);
    void copyFromFull(FloatType *a, int64_t lda);

    void gather();

    Tile<FloatType>* &operator()(int64_t m, int64_t n) {
        return (*tiles_)[std::pair<int64_t, int64_t>(it_+m, jt_+n)];
    }
    Tile<FloatType>* &operator()(int64_t m, int64_t n) const {
        return (*tiles_)[std::pair<int64_t, int64_t>(it_+m, jt_+n)];
    }

    void trsm(blas::Side side, blas::Uplo uplo,
              blas::Op trans, blas::Diag diag,
              FloatType alpha, const Matrix &a);

    void potrf(blas::Uplo uplo, int64_t lookahead=3);

private:
    MPI_Comm mpi_comm_;
    MPI_Group mpi_group_;
    int mpi_size_;
    int mpi_rank_;
    std::function <int64_t (int64_t i, int64_t j)> tileRankFunc;
    std::function <int64_t (int64_t i)> tileMbFunc;
    std::function <int64_t (int64_t j)> tileNbFunc;

    int64_t tileRank(int64_t i, int64_t j) {
        return tileRankFunc(it_+i, jt_+j);
    }
    int64_t tileMb(int64_t i) { return tileMbFunc(it_+i); }
    int64_t tileNb(int64_t j) { return tileNbFunc(jt_+j); }

    bool tileIsLocal(int64_t i, int64_t j) {
        return tileRank(i, j) == mpi_rank_;
    }

    void syrkTask(blas::Uplo uplo, blas::Op trans,
                  FloatType alpha, const Matrix &a, FloatType beta);

    void syrkNest(blas::Uplo uplo, blas::Op trans,
                  FloatType alpha, const Matrix &a, FloatType beta);

    void syrkBatch(blas::Uplo uplo, blas::Op trans,
                   FloatType alpha, const Matrix &a, FloatType beta);

    void tileSend(int64_t i, int64_t j, int dest);
    void tileRecv(int64_t i, int64_t j, int src);
    
    void tileBcast(int64_t m, int64_t n);
    void tileIbcast(int64_t m, int64_t n,
                    std::array<int64_t, 4> range);

    void tileIbcast(int64_t m, int64_t n,
                    std::array<int64_t, 4> range1,
                    std::array<int64_t, 4> range2);

    void tileIbcastFindRanks(int64_t i, int64_t j,
                             std::array<int64_t, 4> range,
                             std::set<int> *bcast_set);

    void tileIbcastIbcast(int64_t i, int64_t j, std::set<int> &bcast_set);
    void tileIbcastIsend(int64_t i, int64_t j, std::set<int> &bcast_set);
    void tileWait(int64_t m, int64_t n);
};

//------------------------------------------------------------------------------
template<typename FloatType>
Matrix<FloatType>::Matrix(int64_t m, int64_t n, double *a, int64_t lda,
                          int64_t mb, int64_t nb)
{
    tiles_ = new std::map<std::pair<int64_t, int64_t>, Tile<FloatType>*>;
    it_ = 0;
    jt_ = 0;
    mt_ = m % mb == 0 ? m/mb : m/mb+1;
    nt_ = n % nb == 0 ? n/nb : n/nb+1;

    tileRankFunc = [] (int64_t i, int64_t j) { return 0; };
    tileMbFunc = [=] (int64_t i) { return (it_+i)*mb > m ? m%mb : mb; };
    tileNbFunc = [=] (int64_t j) { return (jt_+j)*nb > n ? n%nb : nb; };

    copyTo(a, lda);
}

//------------------------------------------------------------------------------
template<typename FloatType>
Matrix<FloatType>::Matrix(int64_t m, int64_t n, double *a, int64_t lda,
                          int64_t mb, int64_t nb,
                          MPI_Comm mpi_comm, int64_t p, int64_t q)
{
    tiles_ = new std::map<std::pair<int64_t, int64_t>, Tile<FloatType>*>;
    it_ = 0;
    jt_ = 0;
    mt_ = m % mb == 0 ? m/mb : m/mb+1;
    nt_ = n % nb == 0 ? n/nb : n/nb+1;

    mpi_comm_ = mpi_comm;
    assert(MPI_Comm_rank(mpi_comm_, &mpi_rank_) == MPI_SUCCESS);
    assert(MPI_Comm_size(mpi_comm_, &mpi_size_) == MPI_SUCCESS);
    assert(MPI_Comm_group(mpi_comm_, &mpi_group_) == MPI_SUCCESS);

    tileRankFunc = [=] (int64_t i, int64_t j) { return i%p + (j%q)*p; };
    tileMbFunc = [=] (int64_t i) { return i*mb > m ? m%mb : mb; };
    tileNbFunc = [=] (int64_t j) { return j*nb > n ? n%nb : nb; };

    copyTo(a, lda);
}

//------------------------------------------------------------------------------
template<typename FloatType>
Matrix<FloatType>::Matrix(const Matrix &a, int64_t it, int64_t jt,
                          int64_t mt, int64_t nt)
{
    assert(it+mt <= a.mt_);
    assert(jt+nt <= a.nt_);
    *this = a;
    it_ += it;
    jt_ += jt;
    mt_ = mt;
    nt_ = nt;
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::copyTo(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i; ++j) {
            if (tileIsLocal(i, j))
                (*this)(i, j) =
                    new Tile<FloatType>(tileMb(i), tileNb(j),
                                        &a[(size_t)lda*n+m], lda);
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::copyFrom(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i; ++j) {
            if (tileIsLocal(i, j)) {
                (*this)(i, j)->copyFrom(&a[(size_t)lda*m+n], lda);
            }
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::copyFromFull(FloatType *a, int64_t lda)
{
    int64_t m = 0;
    for (int64_t i = 0; i < mt_; ++i) {
        int64_t n = 0;
        for (int64_t j = 0; j <= i; ++j) {
            (*this)(i, j)->copyFrom(&a[(size_t)lda*n+m], lda);
            n += tileNb(j);
        }
        m += tileMb(i);
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::gather()
{
    for (int64_t i = 0; i < mt_; ++i) {
        for (int64_t j = 0; j <= i && j < nt_; ++j) {
            if (mpi_rank_ == 0) {
                if (!tileIsLocal(i, j))
                    tileRecv(i, j, tileRank(i, j));
            }
            else {
                if (tileIsLocal(i, j))
                    tileSend(i, j, 0);
            }
        }
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::syrkTask(blas::Uplo uplo, blas::Op trans,
                                 FloatType alpha, const Matrix &that,
                                 FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;
    Matrix<FloatType> a = that;

    // Lower, NoTrans
    for (int64_t n = 0; n < c.nt_; ++n) {

        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            if (c.tileIsLocal(n, n)) {
                a.tileWait(n, k);
                c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
            }

        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                #pragma omp task
                if (c.tileIsLocal(m, n)) {
                    a.tileWait(m, k);
                    a.tileWait(n, k);
                    c(m, n)->gemm(trans, Op::Trans,
                                  alpha, a(m, k), a(n, k), k == 0 ? beta : 1.0);
                }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::syrkNest(blas::Uplo uplo, blas::Op trans,
                                 FloatType alpha, const Matrix &that,
                                 FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;
    Matrix<FloatType> a = that;

    for (int64_t n = 0; n < c.nt_; ++n) {
        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            if (c.tileIsLocal(n, n)) {
                a.tileWait(n, k);
                c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
            }
    }

//  #pragma omp parallel for collapse(3) schedule(dynamic, 1) num_threads(60)
    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int64_t n = 0; n < c.nt_; ++n) {
        for (int64_t m = 0; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (m >= n+1)
                    if (c.tileIsLocal(m, n)) {
                        a.tileWait(m, k);
                        a.tileWait(n, k);
                        c(m, n)->gemm(trans, Op::Trans,
                                      alpha, a(m, k), a(n, k),
                                      k == 0 ? beta : 1.0);
                    }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::syrkBatch(blas::Uplo uplo, blas::Op trans,
                                  FloatType alpha, const Matrix &that,
                                  FloatType beta)
{
    using namespace blas;

    Matrix<FloatType> c = *this;
    Matrix<FloatType> a = that;

    // Lower, NoTrans
    for (int64_t n = 0; n < c.nt_; ++n) {
        for (int64_t k = 0; k < a.nt_; ++k)
            #pragma omp task
            if (c.tileIsLocal(n, n)) {
                a.tileWait(n, k);
                c(n, n)->syrk(uplo, trans, -1.0, a(n, k), k == 0 ? beta : 1.0);
            }
    }

    CBLAS_TRANSPOSE transa_array[1];
    CBLAS_TRANSPOSE transb_array[1];
    int m_array[1];
    int n_array[1];
    int k_array[1];
    double alpha_array[1];
    const double **a_array;
    int lda_array[1];
    const double **b_array;
    int ldb_array[1];
    double beta_array[1];
    double **c_array;
    int ldc_array[1];

    int nb = tileNb(0);
    transa_array[0] = CblasNoTrans;
    transb_array[0] = CblasTrans;
    m_array[0] = nb;
    n_array[0] = nb;
    k_array[0] = nb;
    alpha_array[0] = alpha;
    lda_array[0] = nb;
    ldb_array[0] = nb;
    beta_array[0] = beta;
    ldc_array[0] = nb;

    int group_size = 0;
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (c.tileIsLocal(m, n)) {
                    a.tileWait(m, k);
                    a.tileWait(n, k);
                    ++group_size;
                }

    a_array = (const double**)malloc(sizeof(double*)*group_size);
    b_array = (const double**)malloc(sizeof(double*)*group_size);
    c_array = (double**)malloc(sizeof(double*)*group_size);
    assert(a_array != nullptr);
    assert(b_array != nullptr);
    assert(c_array != nullptr);

    int i = 0;
    for (int64_t n = 0; n < c.nt_; ++n)
        for (int64_t m = n+1; m < c.mt_; ++m)
            for (int64_t k = 0; k < a.nt_; ++k)
                if (c.tileIsLocal(m, n)) {
                    a_array[i] = a(m, k)->data_;
                    b_array[i] = a(n, k)->data_;
                    c_array[i] = c(m, n)->data_;
                    ++i;
                }

    trace_cpu_start();
//  mkl_set_num_threads_local(60);
    cblas_dgemm_batch(CblasColMajor, transa_array, transb_array,
                      m_array, n_array, k_array, alpha_array,
                      a_array, lda_array, b_array, ldb_array, beta_array,
                      c_array, ldc_array, 1, &group_size);
//  mkl_set_num_threads_local(1);
    trace_cpu_stop("DarkGreen");

    free(a_array);
    free(b_array);
    free(c_array);

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::trsm(blas::Side side, blas::Uplo uplo,
                             blas::Op trans, blas::Diag diag,
                             FloatType alpha, const Matrix &a)
{
    using namespace blas;

    Matrix<FloatType> b = *this;

    // Right, Lower, Trans
    for (int64_t k = 0; k < b.nt_; ++k) {

        for (int64_t m = 0; m < b.mt_; ++m) {
            #pragma omp task
            b(m, k)->trsm(side, uplo, trans, diag, 1.0, a(k, k)); 

            for (int64_t n = k+1; n < b.nt_; ++n)
                #pragma omp task
                b(m, n)->gemm(Op::NoTrans, trans,
                              -1.0/alpha, b(m, k), a(n, k), 1.0);
        }
    }
    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileSend(int64_t i, int64_t j, int dest)
{
    Tile<FloatType> *tile = (*this)(i, j);
    int count = tile->mb_*tile->nb_;
    int tag = 0;
    int retval;
    #pragma omp critical
    retval = MPI_Send(tile->data_, count, MPI_DOUBLE, dest, tag, mpi_comm_);
    assert(retval == MPI_SUCCESS);
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileRecv(int64_t i, int64_t j, int src)
{
    Tile<FloatType> *tile = new Tile<FloatType>(tileMb(i), tileNb(j));
    (*this)(i, j) = tile;
    int count = tile->mb_*tile->nb_;
    int tag = 0;
    int retval;
    #pragma omp critical
    retval = MPI_Recv(tile->data_, count, MPI_DOUBLE, src, tag, mpi_comm_,
                      MPI_STATUS_IGNORE);
    assert(retval == MPI_SUCCESS);
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileBcast(int64_t i, int64_t j)
{
    Tile<FloatType> *tile;

    if (tileIsLocal(i, j)) {
        tile = (*this)(i, j);
    }
    else {
        tile = new Tile<FloatType>(tileMb(i), tileNb(j));
        (*this)(i, j) = tile;
    }

    int count = tile->mb_*tile->nb_;
    int retval;
    #pragma omp critical
    retval = MPI_Bcast(tile->data_, count, MPI_DOUBLE, tileRank(i, j),
                       mpi_comm_);
    assert(retval == MPI_SUCCESS);
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileIbcast(int64_t i, int64_t j,
                                   std::array<int64_t, 4> range)
{
    // Find the set of participating ranks.
    std::set<int> bcast_set;
    bcast_set.insert(tileRank(i, j));
    tileIbcastFindRanks(i, j, range, &bcast_set);

    // If contained in the set.
    if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

        // If receiving the tile.
        if (!tileIsLocal(i, j)) {

            // Create the tile.
            Tile<FloatType> *tile;
            tile = new Tile<FloatType>(tileMb(i), tileNb(j));
            (*this)(i, j) = tile;
        }
        // Perform the communication.
        tileIbcastIsend(i, j, bcast_set);
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileIbcast(int64_t i, int64_t j,
                                   std::array<int64_t, 4> range1,
                                   std::array<int64_t, 4> range2)
{
    // Find the set of participating ranks.
    std::set<int> bcast_set;
    bcast_set.insert(tileRank(i, j));
    tileIbcastFindRanks(i, j, range1, &bcast_set);
    tileIbcastFindRanks(i, j, range2, &bcast_set);

    // If contained in the set.
    if (bcast_set.find(mpi_rank_) != bcast_set.end()) {

        // If receiving the tile.
        if (!tileIsLocal(i, j)) {

            // Create the tile.
            Tile<FloatType> *tile;
            tile = new Tile<FloatType>(tileMb(i), tileNb(j));
            (*this)(i, j) = tile;
        }
        // Perform the communication.
        tileIbcastIsend(i, j, bcast_set);
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileIbcastFindRanks(int64_t i, int64_t j,
                                            std::array<int64_t, 4> range,
                                            std::set<int> *bcast_set)
{
    int64_t i1 = range[0];
    int64_t i2 = range[1];
    int64_t j1 = range[2];
    int64_t j2 = range[3];

    // Find the set of participating ranks.
    for (int64_t i = i1; i <= i2; ++i)
        for (int64_t j = j1; j <= j2; ++j)
            bcast_set->insert(tileRank(i, j));
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileIbcastIbcast(
    int64_t i, int64_t j, std::set<int> &bcast_set)
{
    // Convert the set of ranks to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // Create the broadcast group.
    Tile<FloatType> *tile = (*this)(i, j);
    int retval;
    #pragma omp critical
    retval = MPI_Group_incl(mpi_group_, bcast_vec.size(), bcast_vec.data(),
                            &tile->bcast_group_);
    assert(retval == MPI_SUCCESS);

    // Create a broadcast communicator.
    int tag = 0;
    trace_cpu_start();
    #pragma omp critical
    retval = MPI_Comm_create_group(mpi_comm_, tile->bcast_group_, tag,
                                   &tile->bcast_comm_);
    assert(retval == MPI_SUCCESS);
    assert(tile->bcast_comm_ != MPI_COMM_NULL);
    trace_cpu_stop("Crimson");

    // Find the broadcast rank.
    int bcast_rank;
    #pragma omp critical
    MPI_Comm_rank(tile->bcast_comm_, &bcast_rank);

    // Find the broadcast root rank.
    int root_rank = tileRank(i, j);
    int bcast_root;
    #pragma omp critical
    retval = MPI_Group_translate_ranks(mpi_group_, 1, &root_rank,
                                       tile->bcast_group_, &bcast_root);
    assert(retval == MPI_SUCCESS);

    // Do the broadcast.
    int count = tile->mb_*tile->nb_;
    #pragma omp critical
    retval = MPI_Ibcast(tile->data_, count, MPI_DOUBLE, bcast_root,
                        tile->bcast_comm_, &tile->bcast_request_);
    assert(retval == MPI_SUCCESS);

    // Free the group.
    // #pragma omp critical
    // retval = MPI_Group_free(&tile->bcast_group_);
    // assert(retval == MPI_SUCCESS);

    // Free the communicator.
    // #pragma omp critical
    // retval = MPI_Comm_free(&tile->bcast_comm_);
    // assert(retval == MPI_SUCCESS);
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileIbcastIsend(
    int64_t i, int64_t j, std::set<int> &bcast_set)
{
    // If sending the tile.
    if (tileIsLocal(i, j)) {

        // For each rank in the bcast.
        std::for_each(bcast_set.begin(), bcast_set.end(), [&](const int &rank) {

            // If not my own rank.
            if (rank != mpi_rank_) {

                // Send the tile.
                Tile<FloatType> *tile = (*this)(i, j);
                int count = tile->mb_*tile->nb_;
                int dst = rank;
                int tag = 0;

                trace_cpu_start();
                int retval;
                #pragma omp critical
                retval = MPI_Isend(tile->data_, count, MPI_DOUBLE, dst, tag,
                                   mpi_comm_, &tile->bcast_request_);
                assert(retval == MPI_SUCCESS);
                #pragma omp critical
                retval = MPI_Request_free(&tile->bcast_request_);
                assert(retval == MPI_SUCCESS);
                trace_cpu_stop("Salmon");
            }
        });
    }
    else {

        // Receive the tile.
        Tile<FloatType> *tile = (*this)(i, j);
        int count = tile->mb_*tile->nb_;
        int src = tileRank(i, j);
        int tag = 0;

        trace_cpu_start();
        int retval;
        #pragma omp critical
        retval = MPI_Irecv(tile->data_, count, MPI_DOUBLE, src, tag, mpi_comm_,
                           &tile->bcast_request_);
        assert(retval == MPI_SUCCESS);
        trace_cpu_stop("Crimson");
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::tileWait(int64_t i, int64_t j)
{
    if (!tileIsLocal(i, j)) {
        Tile<FloatType> *tile = (*this)(i, j);

        trace_cpu_start();
        int retval;
        #pragma omp critical
        retval = MPI_Wait(&tile->bcast_request_, MPI_STATUS_IGNORE);
        assert(retval == MPI_SUCCESS);
        trace_cpu_stop("DarkRed");
    }
}

//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::potrf(blas::Uplo uplo, int64_t lookahead)
{
    using namespace blas;

    Matrix<FloatType> a = *this;
    uint8_t *column;

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < nt_; ++k) {
        // panel
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            if (tileIsLocal(k, k))
                a(k, k)->potrf(uplo);

            if (k < nt_-1)
                tileIbcast(k, k, {k+1, nt_-1, k, k});

            for (int64_t m = k+1; m < nt_; ++m) {

                if (tileIsLocal(m, k)) {
                    tileWait(k, k);
                    a(m, k)->trsm(Side::Right, Uplo::Lower,
                                  Op::Trans, Diag::NonUnit,
                                  1.0, a(k, k));
                }
                tileIbcast(m, k, {m, m, k+1, m},
                                 {m, nt_-1, m, m});
            }
            #pragma omp taskwait
        }
        // lookahead column(s)
        for (int64_t n = k+1; n < k+1+lookahead && n < nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n]) priority(1)
            {
                #pragma omp task priority(1)
                if (tileIsLocal(n, n)) {
                    tileWait(n, k);
                    a(n, n)->syrk(Uplo::Lower, Op::NoTrans,
                                  -1.0, a(n, k), 1.0);
                }

                for (int64_t m = n+1; m < nt_; ++m) {
                    #pragma omp task priority(1)
                    if (tileIsLocal(m, n)) {
                        tileWait(m, k);
                        tileWait(n, k);
                        a(m, n)->gemm(Op::NoTrans, Op::Trans,
                                      -1.0, a(m, k), a(n, k), 1.0);
                    }
                }
                #pragma omp taskwait
            }
        }
        // trailing submatrix
        if (k+1+lookahead < nt_) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[nt_-1])
            Matrix(a, k+1+lookahead, k+1+lookahead,
                   nt_-1-k-lookahead, nt_-1-k-lookahead).syrkNest(
                Uplo::Lower, Op::NoTrans,
                -1.0, Matrix(a, k+1+lookahead, k, nt_-1-k-lookahead, 1), 1.0);
        }
    }
}
/*
//------------------------------------------------------------------------------
template<typename FloatType>
void Matrix<FloatType>::potrf(blas::Uplo uplo, int64_t lookahead)
{
    using namespace blas;

    Matrix<FloatType> a = *this;
    uint8_t *column;

//  #pragma omp parallel num_threads(8)
    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < nt_; ++k) {
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            a(k, k)->potrf(uplo);

            for (int64_t m = k+1; m < nt_; ++m) {
                #pragma omp task priority(1)
                {
                    a(m, k)->trsm(Side::Right, Uplo::Lower,
                                  Op::Trans, Diag::NonUnit,
                                  1.0, a(k, k));

                    if (m-k-1 > 0)
                        a(m, k)->packA(m-k-1);

                    if (nt_-m-1 > 0)
                        a(m, k)->packB(nt_-m-1);
                }
            }
            #pragma omp taskwait
        }
        for (int64_t n = k+1; n < k+1+lookahead && n < nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n]) priority(1)
            {
                #pragma omp task priority(1)
                a(n, n)->syrk(Uplo::Lower, Op::NoTrans,
                              -1.0, a(n, k), 1.0);

                for (int64_t m = n+1; m < nt_; ++m) {
                    #pragma omp task priority(1)
                    a(m, n)->gemm(Op::NoTrans, Op::Trans,
                                  -1.0, a(m, k), a(n, k), 1.0);
                }
                #pragma omp taskwait
            }
        }
        if (k+1+lookahead < nt_)
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[nt_-1])
            Matrix(a, k+1+lookahead, k+1+lookahead,
                   nt_-1-k-lookahead, nt_-1-k-lookahead).syrkNest(
                Uplo::Lower, Op::NoTrans,
                -1.0, Matrix(a, k+1+lookahead, k, nt_-1-k-lookahead, 1), 1.0);
    }
}
*/
} // namespace slate

#endif // SLATE_MATRIX_HH