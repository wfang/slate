
#include "blas.hh"

namespace blas {

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          float alpha, float *a, int64_t lda,
                       float *b, int64_t ldb)
{
    cblas_strsm(CblasColMajor,
                CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                m, n,
                alpha, a, lda,
                       b, ldb);
}

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          double alpha, double *a, int64_t lda,
                        double *b, int64_t ldb)
{
    cblas_dtrsm(CblasColMajor,
                CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                m, n,
                alpha, a, lda,
                       b, ldb);
}

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          std::complex<float> alpha, std::complex<float> *a, int64_t lda,
                                     std::complex<float> *b, int64_t ldb)
{
    cblas_ctrsm(CblasColMajor,
                CblasRight, CblasLower, CblasConjTrans, CblasNonUnit,
                m, n,
                (const void*)&alpha, a, lda,
                                     b, ldb);
}

void trsm(blas::Layout layout,
          blas::Side side, blas::Uplo uplo, blas::Op transa, blas::Diag diag,
          int64_t m, int64_t n,
          std::complex<double> alpha, std::complex<double> *a, int64_t lda,
                                      std::complex<double> *b, int64_t ldb)
{
    cblas_ztrsm(CblasColMajor,
                CblasRight, CblasLower, CblasConjTrans, CblasNonUnit,
                m, n,
                (const void*)&alpha, a, lda,
                                     b, ldb);
}

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          float alpha, float *a, int64_t lda,
          float beta,  float *c, int64_t ldc)
{
    cblas_ssyrk(CblasColMajor,
                CblasLower, CblasNoTrans,
                n, k,
                alpha, a, lda,
                beta,  c, ldc);
}

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          double alpha, double *a, int64_t lda,
          double beta,  double *c, int64_t ldc)
{
    cblas_dsyrk(CblasColMajor,
                CblasLower, CblasNoTrans,
                n, k,
                alpha, a, lda,
                beta,  c, ldc);
}

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          float alpha, std::complex<float> *a, int64_t lda,
          float beta,  std::complex<float> *c, int64_t ldc)
{
    cblas_cherk(CblasColMajor,
                CblasLower, CblasNoTrans,
                n, k,
                alpha, a, lda,
                beta,  c, ldc);
}

void syrk(blas::Layout layout,
          blas::Uplo uplo, blas::Op trans,
          int64_t n, int64_t k,
          double alpha, std::complex<double> *a, int64_t lda,
          double beta,  std::complex<double> *c, int64_t ldc)
{
    cblas_zherk(CblasColMajor,
                CblasLower, CblasNoTrans,
                n, k,
                alpha, a, lda,
                beta,  c, ldc);
}

void gemm(blas::Layout layout,
         blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          float alpha, float *a, int64_t lda,
                       float *b, int64_t ldb,
          float beta,  float *c, int64_t ldc)
{
    cblas_sgemm(CblasColMajor,
                CblasNoTrans, CblasTrans,
                m, n, k, 
                alpha, a, lda,
                       b, ldb,
                beta,  c, ldc);
}


void gemm(blas::Layout layout,
          blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          double alpha, double *a, int64_t lda,
                        double *b, int64_t ldb,
          double beta,  double *c, int64_t ldc)
{
    if (transa == blas::Op::NoTrans && transb == blas::Op::Trans) {
	cblas_dgemm(CblasColMajor,
		    CblasNoTrans, CblasTrans,
		    m, n, k, 
		    alpha, a, lda,
		    b, ldb,
		    beta,  c, ldc);
    } else if (transa == blas::Op::NoTrans && transb == blas::Op::NoTrans) {
	cblas_dgemm(CblasColMajor,
		    CblasNoTrans, CblasNoTrans,
		    m, n, k, 
		    alpha, a, lda,
		    b, ldb,
		    beta,  c, ldc);
    } else {
	// Not implemented!
	assert(0);
    }
	
}

void gemm(blas::Layout layout,
          blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          std::complex<float> alpha, std::complex<float> *a, int64_t lda,
                                     std::complex<float> *b, int64_t ldb,
          std::complex<float> beta,  std::complex<float> *c, int64_t ldc)
{
    cblas_cgemm(CblasColMajor,
                CblasNoTrans, CblasTrans,
                m, n, k, 
                (const void*)&alpha, a, lda,
                                     b, ldb,
                (const void*)&beta,  c, ldc);
}

void gemm(blas::Layout layout,
          blas::Op transa, blas::Op transb,
          int64_t m, int64_t n, int64_t k,
          std::complex<double> alpha, std::complex<double> *a, int64_t lda,
                                      std::complex<double> *b, int64_t ldb,
          std::complex<double> beta,  std::complex<double> *c, int64_t ldc)
{
    cblas_zgemm(CblasColMajor,
                CblasNoTrans, CblasTrans,
                m, n, k, 
                (const void*)&alpha, a, lda,
                                     b, ldb,
                (const void*)&beta,  c, ldc);
}

} // namespace blas
