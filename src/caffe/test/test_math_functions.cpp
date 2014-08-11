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

TYPED_TEST_CASE(MathFunctionsTest, TestDtypes);

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

TYPED_TEST(MathFunctionsTest, TestSignCPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sign<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
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

TYPED_TEST(MathFunctionsTest, TestFabsCPU) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_fabs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
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

TYPED_TEST(MathFunctionsTest, TestCopyCPU) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  TypeParam* top_data = this->blob_top_->mutable_cpu_data();
  Caffe::set_mode(Caffe::CPU);
  caffe_copy(n, bottom_data, top_data);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#ifndef CPU_ONLY

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

TYPED_TEST(MathFunctionsTest, TestCopyGPU) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  Caffe::set_mode(Caffe::GPU);
  caffe_copy(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}
#endif

template<typename Dtype>
class CsrFunctionsGenTest : public ::testing::Test {
 protected:
	CsrFunctionsGenTest(): A_(), indices_(), ptr_(),B_(), C_(), NZZ(0), M(0), N(0), K(0), TransA(CblasNoTrans), TransB(CblasNoTrans), alpha(1.0), beta(0.0),orderC(CblasRowMajor) {
	}

  virtual void SetUp(int m, int n, int k, int nzz, int ptr_size) {
	  M = m;
	  N = n;
	  K = k;
	  NZZ = nzz;
	  PTR_SIZE = ptr_size;

	  A_.reset(new SyncedMemory(nzz * sizeof(Dtype)));
	  indices_.reset(new SyncedMemory(nzz * sizeof(int)));
	  ptr_.reset(new SyncedMemory(ptr_size * sizeof(int)));
	  B_.reset(new SyncedMemory(K * N * sizeof(Dtype)));
	  C_.reset(new SyncedMemory(M * N * sizeof(Dtype)));
  }

  virtual void run(bool isCpu){
	  if (isCpu){
		  caffe_cpu_csr_gemm(TransA,TransB,M,N,K, alpha ,NZZ , cpu_A(), cpu_indices(), cpu_ptr(),cpu_B(), beta,cpu_C(), orderC);
	  }else{
#ifndef CPU_ONLY
		  caffe_gpu_csr_gemm(TransA,TransB,M,N,K, alpha ,NZZ , gpu_A(), gpu_indices(), gpu_ptr(),gpu_B(), beta,gpu_C(), orderC);
#else

#endif
	  }
  }

  void setA(Dtype A_data[],int A_indices[], int A_ptr[] ){
	  Dtype* am = cpu_A();
	  int* aindices = cpu_indices();
	  int* aptr = cpu_ptr();

	  for (int i=0; i < NZZ; i++){
		  am[i] = A_data[i];
		  aindices[i] = A_indices[i];
	  }
	  for (int i=0; i < PTR_SIZE; i++){
		  aptr[i] = A_ptr[i];
	  }
  }

  void setB(Dtype B_data[] ){
  	  Dtype* bm = cpu_B();
  	  for (int i=0; i < (K*N); i++){
  		  bm[i] = B_data[i];
  	  }
    }
  void setC(Dtype C_data[] ){
    	  Dtype* cm = cpu_C();
    	  for (int i=0; i < (M*N); i++){
    		  cm[i] = C_data[i];
    }
  }
 void checkC(Dtype C_check[]){
	 Dtype* cm = cpu_C();
	 for (int i=0; i < (M*N); i++){
	    EXPECT_EQ(cm[i], C_check[i]);
	 }
 }

  Dtype* cpu_A() {
    CHECK(A_);
    return reinterpret_cast<Dtype*>(A_->mutable_cpu_data());
  }
  Dtype* gpu_A() {
      CHECK(A_);
      return reinterpret_cast<Dtype*>(A_->mutable_gpu_data());
    }

  Dtype* cpu_B() {
     CHECK(B_);
     return reinterpret_cast<Dtype*>(B_->mutable_cpu_data());
   }
  Dtype* gpu_B() {
      CHECK(B_);
      return reinterpret_cast<Dtype*>(B_->mutable_gpu_data());
    }

  Dtype* cpu_C() {
     CHECK(C_);
     return reinterpret_cast<Dtype*>(C_->mutable_cpu_data());
  }
  Dtype* gpu_C() {
       CHECK(C_);
       return reinterpret_cast<Dtype*>(C_->mutable_gpu_data());
    }

  int* cpu_indices() {
       CHECK(indices_);
       return reinterpret_cast<int*>(indices_->mutable_cpu_data());
  }
  int* gpu_indices() {
         CHECK(indices_);
         return reinterpret_cast<int*>(indices_->mutable_gpu_data());
    }

  int* cpu_ptr() {
         CHECK(ptr_);
         return reinterpret_cast<int*>(ptr_->mutable_cpu_data());
  }
  int* gpu_ptr() {
           CHECK(ptr_);
           return reinterpret_cast<int*>(ptr_->mutable_gpu_data());
    }

  shared_ptr<SyncedMemory> A_;
  shared_ptr<SyncedMemory> indices_;
  shared_ptr<SyncedMemory> ptr_;
  shared_ptr<SyncedMemory> B_;
  shared_ptr<SyncedMemory> C_;
  int M;
  int N;
  int K;
  int NZZ;
  int PTR_SIZE;

  CBLAS_TRANSPOSE TransA;
  CBLAS_TRANSPOSE TransB;
  Dtype alpha;
  Dtype beta;
  CBLAS_ORDER orderC;

};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(CsrFunctionsGenTest, Dtypes);

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm1) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {0.0,0.0,0.0,0.0};
	TypeParam CCheck[] = {16.0,25.0,15.0,24.0};
	this->alpha = 1.0;
	this->beta = 1.0;
	this->SetUp(2,2,3,3,3);


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);

}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm2) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,4.0};
	TypeParam CCheck[] = {17.0,27.0,18.0,28.0};
	this->alpha = 1.0;
	this->beta = 1.0;
	this->SetUp(2,2,3,3,3);
	this->TransA = CblasNoTrans;
	this->TransB = CblasNoTrans;
	this->orderC = CblasRowMajor;


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm3) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.3,3.0,4.0};
	TypeParam CCheck[] = {16.0,25.0,15.0,24.0};
	this->alpha = 1.0;
	this->beta = 0.0;
	this->SetUp(2,2,3,3,3);


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm4) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {0.0,0.0,0.0,0.0};
	TypeParam CCheck[] = {16.0,15.0,25.0,24.0};
	this->alpha = 1.0;
	this->beta = 1.0;
	this->SetUp(2,2,3,3,3);
	this->TransA = CblasNoTrans;
	this->TransB = CblasNoTrans;
	this->orderC = CblasColMajor;


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm5) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,0.0,0.0};
	TypeParam CCheck[] = {17.0,17.0,25.0,24.0};
	this->alpha = 1.0;
	this->beta = 1.0;
	this->SetUp(2,2,3,3,3);
	this->TransA = CblasNoTrans;
	this->TransB = CblasNoTrans;
	this->orderC = CblasColMajor;


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm6) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,0.0};
	TypeParam CCheck[] = {16.0,15.0,25.0,24.0};
	this->alpha = 1.0;
	this->beta = 0.0;
	this->SetUp(2,2,3,3,3);
	this->TransA = CblasNoTrans;
	this->TransB = CblasNoTrans;
	this->orderC = CblasColMajor;


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm7) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {0.0,0.0,0.0,0.0};
	TypeParam CCheck[] = {32.0,50.0,30.0,48.0};
	this->alpha = 2.0;
	this->beta = 1.0;
	this->SetUp(2,2,3,3,3);


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);

}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm8) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,0.0};
	TypeParam CCheck[] = {31.0,58.0,51.0,36.0};
	this->alpha = 2.0;
	this->beta = 3.0;
	this->SetUp(2,2,3,3,3);
	this->TransA = CblasNoTrans;
	this->TransB = CblasTrans;
	this->orderC = CblasRowMajor;


	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm9) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,0.0};
	TypeParam CCheck[] = {31.0,48.0,61.0,36.0};
	this->alpha = 2.0;
	this->beta = 3.0;
	this->SetUp(2,2,3,3,3);
	this->TransA = CblasNoTrans;
	this->TransB = CblasTrans;
	this->orderC = CblasColMajor;

	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm10) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0};
	TypeParam CCheck[] = {11.0,20.0,19.0,48.0,36.0,54.0,16.0,28.0,20.0};
	this->alpha = 2.0;
	this->beta = 3.0;
	this->SetUp(3,3,2,3,3);
	this->TransA = CblasTrans;
	this->TransB = CblasNoTrans;
	this->orderC = CblasRowMajor;

	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}


TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm11) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0};
	TypeParam CCheck[] = {11.0,54.0,25.0,14.0,36.0,28.0,10.0,54.0,20.0};
	this->alpha = 2.0;
	this->beta = 3.0;
	this->SetUp(3,3,2,3,3);
	this->TransA = CblasTrans;
	this->TransB = CblasNoTrans;
	this->orderC = CblasColMajor;

	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}


TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm12) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0};
	TypeParam CCheck[] = {11.0,16.0,21.0,42.0,48.0,54.0,16.0,20.0,24.0};
	this->alpha = 2.0;
	this->beta = 3.0;
	this->SetUp(3,3,2,3,3);
	this->TransA = CblasTrans;
	this->TransB = CblasTrans;
	this->orderC = CblasRowMajor;

	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}

TYPED_TEST(CsrFunctionsGenTest, TestCsrGemm13) {

	TypeParam A[] = {1.0,2.0,3.0};
	int indices[] = {0,2,1};
	int ptr[] = {0,2,3};
	TypeParam B[] = {4.0,7.0,5.0,8.0,6.0,9.0};
	TypeParam C[] = {1.0,2.0,3.0,0.0,0.0,0.0,0.0,0.0,0.0};
	TypeParam CCheck[] = {11.0,48.0,25.0,10.0,48.0,20.0,12.0,54.0,24.0};
	this->alpha = 2.0;
	this->beta = 3.0;
	this->SetUp(3,3,2,3,3);
	this->TransA = CblasTrans;
	this->TransB = CblasTrans;
	this->orderC = CblasColMajor;

	this->setA(A,indices,ptr);
	this->setB(B);
	this->setC(C);
	this->run(true);
	this->checkC(CCheck);
	this->setC(C);
	this->run(false);
	this->checkC(CCheck);
}







}  // namespace caffe
