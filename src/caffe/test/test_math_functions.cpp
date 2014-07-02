// Copyright 2014 BVLC and contributors.

#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <climits>
#include <cmath>  // for std::fabs
#include <cstdlib>  // for rand_r

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename Dtype>
class MathFunctionsTest : public ::testing::Test {
 protected:
  MathFunctionsTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(11, 17, 19, 23);
    this->blob_top_->Reshape(11, 17, 19, 23);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~MathFunctionsTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  // http://en.wikipedia.org/wiki/Hamming_distance
  int ReferenceHammingDistance(const int n, const Dtype* x, const Dtype* y) {
    int dist = 0;
    uint64_t val;
    for (int i = 0; i < n; ++i) {
      if (sizeof(Dtype) == 8) {
        val = static_cast<uint64_t>(x[i]) ^ static_cast<uint64_t>(y[i]);
      } else if (sizeof(Dtype) == 4) {
        val = static_cast<uint32_t>(x[i]) ^ static_cast<uint32_t>(y[i]);
      } else {
        LOG(FATAL) << "Unrecognized Dtype size: " << sizeof(Dtype);
      }
      // Count the number of set bits
      while (val) {
        ++dist;
        val &= val - 1;
      }
    }
    return dist;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MathFunctionsTest, Dtypes);

TYPED_TEST(MathFunctionsTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

TYPED_TEST(MathFunctionsTest, TestHammingDistanceCPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  EXPECT_EQ(this->ReferenceHammingDistance(n, x, y),
            caffe_cpu_hamming_distance<TypeParam>(n, x, y));
}

// TODO: Fix caffe_gpu_hamming_distance and re-enable this test.
TYPED_TEST(MathFunctionsTest, DISABLED_TestHammingDistanceGPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  int reference_distance = this->ReferenceHammingDistance(n, x, y);
  x = this->blob_bottom_->gpu_data();
  y = this->blob_top_->gpu_data();
  int computed_distance = caffe_gpu_hamming_distance<TypeParam>(n, x, y);
  EXPECT_EQ(reference_distance, computed_distance);
}

TYPED_TEST(MathFunctionsTest, TestAsumCPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam cpu_asum = caffe_cpu_asum<TypeParam>(n, x);
  EXPECT_LT((cpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(MathFunctionsTest, TestAsumGPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam gpu_asum;
  caffe_gpu_asum<TypeParam>(n, this->blob_bottom_->gpu_data(), &gpu_asum);
  EXPECT_LT((gpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(MathFunctionsTest, TestSignCPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sign<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(MathFunctionsTest, TestSignGPU) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sign<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(MathFunctionsTest, TestSgnbitCPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sgnbit<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(MathFunctionsTest, TestSgnbitGPU) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sgnbit<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(MathFunctionsTest, TestFabsCPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_fabs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(MathFunctionsTest, TestFabsGPU) {
  int n = this->blob_bottom_->count();
  caffe_gpu_fabs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(MathFunctionsTest, TestScaleCPU) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_cpu_scale<TypeParam>(n, alpha, this->blob_bottom_->cpu_data(),
                             this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(MathFunctionsTest, TestScaleGPU) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_gpu_scale<TypeParam>(n, alpha, this->blob_bottom_->gpu_data(),
                             this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(MathFunctionsTest, TestCopyCPU) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  TypeParam* top_data = this->blob_top_->mutable_cpu_data();
  caffe_copy(n, bottom_data, top_data);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(MathFunctionsTest, TestCopyGPU) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  caffe_gpu_copy(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

template<typename Dtype>
class CsrCpuFunctionsTest : public ::testing::Test {
 protected:
	CsrCpuFunctionsTest() {
		A = (Dtype*) malloc(sizeof(Dtype) * 3);
		indices = (int*) malloc(sizeof(int) * 3);
		ptr = (int*) malloc(sizeof(int) * 3);
		B = (Dtype*) malloc(sizeof(Dtype) * 6);
	}

  virtual void SetUp() {
	  A[0] = 1.0;
	  A[1] = 2.0;
	  A[2] = 3.0;

	  indices[0] = 0;
	  indices[1] = 2;
	  indices[2] = 1;

	  ptr[0] = 0;
	  ptr[1] = 2;
	  ptr[2] = 3;

	  B[0] = 4.0;
	  B[1] = 7.0;
	  B[2] = 5.0;
	  B[3] = 8.0;
	  B[4] = 6.0;
	  B[5] = 9.0;
  }

  virtual ~CsrCpuFunctionsTest() {
	  delete A;
	  delete indices;
	  delete ptr;
	  delete B;
  }
  Dtype* A;
  int* indices;
  int* ptr;
  Dtype* B;

};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(CsrCpuFunctionsTest, Dtypes);


TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm1) {
	TypeParam Ct[] = {(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasNoTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasRowMajor);

	EXPECT_EQ(C[0], (TypeParam)16.0);
	EXPECT_EQ(C[1], (TypeParam)25.0);
	EXPECT_EQ(C[2], (TypeParam)15.0);
	EXPECT_EQ(C[3], (TypeParam)24.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm2) {
	TypeParam Ct[] = {(TypeParam)1.0,(TypeParam)2.0,(TypeParam)3.0,(TypeParam)4.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasNoTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasRowMajor);

	EXPECT_EQ(C[0], (TypeParam)17.0);
	EXPECT_EQ(C[1], (TypeParam)27.0);
	EXPECT_EQ(C[2], (TypeParam)18.0);
	EXPECT_EQ(C[3], (TypeParam)28.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm3) {
	TypeParam Ct[] = {(TypeParam)1.0,(TypeParam)2.0,(TypeParam)3.0,(TypeParam)4.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasNoTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)0.0,C, CblasRowMajor);

	EXPECT_EQ(C[0], (TypeParam)16.0);
	EXPECT_EQ(C[1], (TypeParam)25.0);
	EXPECT_EQ(C[2], (TypeParam)15.0);
	EXPECT_EQ(C[3], (TypeParam)24.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm4) {
	TypeParam Ct[] = {(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasNoTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasColMajor);

	EXPECT_EQ(C[0], (TypeParam)16.0);
	EXPECT_EQ(C[1], (TypeParam)15.0);
	EXPECT_EQ(C[2], (TypeParam)25.0);
	EXPECT_EQ(C[3], (TypeParam)24.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm5) {
	TypeParam Ct[] = {(TypeParam)1.0,(TypeParam)2.0,(TypeParam)3.0,(TypeParam)4.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasNoTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasColMajor);

	EXPECT_EQ(C[0], (TypeParam)17.0);
	EXPECT_EQ(C[1], (TypeParam)17.0);
	EXPECT_EQ(C[2], (TypeParam)28.0);
	EXPECT_EQ(C[3], (TypeParam)28.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm6) {
	TypeParam Ct[] = {(TypeParam)1.0,(TypeParam)2.0,(TypeParam)3.0,(TypeParam)4.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasNoTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)0.0,C, CblasColMajor);

	EXPECT_EQ(C[0], (TypeParam)16.0);
	EXPECT_EQ(C[1], (TypeParam)15.0);
	EXPECT_EQ(C[2], (TypeParam)25.0);
	EXPECT_EQ(C[3], (TypeParam)24.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm7) {
	TypeParam Ct[] = {(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasNoTrans,2,2,3, (TypeParam)2.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasRowMajor);

	EXPECT_EQ(C[0], (TypeParam)32.0);
	EXPECT_EQ(C[1], (TypeParam)50.0);
	EXPECT_EQ(C[2], (TypeParam)30.0);
	EXPECT_EQ(C[3], (TypeParam)48.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm8) {
	TypeParam Ct[] = {(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasRowMajor);

	EXPECT_EQ(C[0], (TypeParam)14.0);
	EXPECT_EQ(C[1], (TypeParam)26.0);
	EXPECT_EQ(C[2], (TypeParam)21.0);
	EXPECT_EQ(C[3], (TypeParam)18.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm9) {
	TypeParam Ct[] = {(TypeParam)1.0,(TypeParam)2.0,(TypeParam)3.0,(TypeParam)4.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasRowMajor);

	EXPECT_EQ(C[0], (TypeParam)15.0);
	EXPECT_EQ(C[1], (TypeParam)28.0);
	EXPECT_EQ(C[2], (TypeParam)24.0);
	EXPECT_EQ(C[3], (TypeParam)22.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm10) {
	TypeParam Ct[] = {(TypeParam)1.0,(TypeParam)2.0,(TypeParam)3.0,(TypeParam)4.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)0.0,C, CblasRowMajor);

	EXPECT_EQ(C[0], (TypeParam)14.0);
	EXPECT_EQ(C[1], (TypeParam)26.0);
	EXPECT_EQ(C[2], (TypeParam)21.0);
	EXPECT_EQ(C[3], (TypeParam)18.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm11) {
	TypeParam Ct[] = {(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0,(TypeParam)0.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasColMajor);

	EXPECT_EQ(C[0], (TypeParam)14.0);
	EXPECT_EQ(C[1], (TypeParam)21.0);
	EXPECT_EQ(C[2], (TypeParam)26.0);
	EXPECT_EQ(C[3], (TypeParam)18.0);
}

TYPED_TEST(CsrCpuFunctionsTest, TestCpuCsrGemm12) {
	TypeParam Ct[] = {(TypeParam)1.0,(TypeParam)2.0,(TypeParam)3.0,(TypeParam)4.0};
	TypeParam* C = Ct;
	caffe_cpu_csr_gemm(CblasNoTrans,CblasTrans,2,2,3, (TypeParam)1.0 ,3 , this->A, this->indices, this->ptr,this->B,(TypeParam)1.0,C, CblasColMajor);

	EXPECT_EQ(C[0], (TypeParam)15.0);
	EXPECT_EQ(C[1], (TypeParam)23.0);
	EXPECT_EQ(C[2], (TypeParam)29.0);
	EXPECT_EQ(C[3], (TypeParam)22.0);
}








}  // namespace caffe
