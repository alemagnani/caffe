// Copyright 2014 BVLC and contributors.

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#define THREADS_PER_BLOCK_CSR 32

namespace caffe {

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = alpha;
	}
}

template <>
void caffe_gpu_set(const int N, const float alpha, float* Y) {
	if (alpha == 0) {
		CUDA_CHECK(cudaMemset(Y, 0, sizeof(float) * N));
		return;
	}
	// NOLINT_NEXT_LINE(whitespace/operators)
	set_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, alpha, Y);
}

template <>
void caffe_gpu_set(const int N, const double alpha, double* Y) {
	if (alpha == 0) {
		CUDA_CHECK(cudaMemset(Y, 0, sizeof(double) * N));
		return;
	}
	// NOLINT_NEXT_LINE(whitespace/operators)
	set_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, alpha, Y);
}

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] += alpha;
	}
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, alpha, Y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
		const Dtype* b, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = a[index] * b[index];
	}
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
		const float* b, float* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
		const double* b, double* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
		const Dtype* b, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = a[index] / b[index];
	}
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
		const float* b, float* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
		const double* b, double* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, b, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
		const Dtype alpha, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = pow(a[index], alpha);
	}
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
		const float alpha, float* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
		const double alpha, double* y) {
	// NOLINT_NEXT_LINE(whitespace/operators)
	powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
			N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
		- (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(fabs, y[index] = fabs(x[index]));

__global__ void popc_kernel(const int n, const float* a,
		const float* b, uint8_t* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = __popc(static_cast<uint32_t>(a[index]) ^
				static_cast<uint32_t>(b[index]));
	}
}

__global__ void popcll_kernel(const int n, const double* a,
		const double* b, uint8_t* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index] = __popcll(static_cast<uint64_t>(a[index]) ^
				static_cast<uint64_t>(b[index]));
	}
}

template <>
uint32_t caffe_gpu_hamming_distance<float>(const int n, const float* x,
		const float* y) {
	// TODO: Fix caffe_gpu_hamming_distance (see failing unit test
	// TestHammingDistanceGPU in test_math_functions.cpp).
	NOT_IMPLEMENTED;
	thrust::device_vector<uint8_t> popcounts(n);
	// NOLINT_NEXT_LINE(whitespace/operators)
	popc_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
			n, x, y, thrust::raw_pointer_cast(popcounts.data()));
	return thrust::reduce(popcounts.begin(), popcounts.end(),
			(uint32_t) 0, thrust::plus<uint32_t>());
}

template <>
uint32_t caffe_gpu_hamming_distance<double>(const int n, const double* x,
		const double* y) {
	// TODO: Fix caffe_gpu_hamming_distance (see failing unit test
	// TestHammingDistanceGPU in test_math_functions.cpp).
	NOT_IMPLEMENTED;
	thrust::device_vector<uint8_t> popcounts(n);
	// NOLINT_NEXT_LINE(whitespace/operators)
	popcll_kernel<<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
			n, x, y, thrust::raw_pointer_cast(popcounts.data()));
	return thrust::reduce(popcounts.begin(), popcounts.end(),
			/* NOLINT_NEXT_LINE(build/include_what_you_use) */
			(uint32_t) 0, thrust::plus<uint32_t>());
}

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
	CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
		float* r) {
	CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
	const float range = b - a;
	if (range != static_cast<float>(1)) {
		caffe_gpu_scal(n, range, r);
	}
	if (a != static_cast<float>(0)) {
		caffe_gpu_add_scalar(n, a, r);
	}
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
		double* r) {
	CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
	const double range = b - a;
	if (range != static_cast<double>(1)) {
		caffe_gpu_scal(n, range, r);
	}
	if (a != static_cast<double>(0)) {
		caffe_gpu_add_scalar(n, a, r);
	}
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
		float* r) {
	CURAND_CHECK(
			curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
		double* r) {
	CURAND_CHECK(
			curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

template <typename Dtype>
__device__ void caffe_gpu_csr_gemm_kernel_core(const int M, const int N, const int K,
		const Dtype alpha, int nzz, const Dtype* A, const int* indices,const int* ptr, const Dtype* B, const int ldb1, const int ldb2, const Dtype beta, Dtype* C, const int ldc1, const int ldc2){

	__shared__ volatile Dtype sums[THREADS_PER_BLOCK_CSR * 2];

	for( int rowA = blockIdx.x; rowA < M; rowA += gridDim.x ){
		const int begin = ptr[rowA];
		const int end = ptr[rowA+1];
		const int offset_c_part = rowA * ldc1;
		for( int colC = blockIdx.y;  colC < N; colC += gridDim.y ){
			Dtype sum = 0.0;
			const int offset_b_part = colC * ldb2;
			for(int pos = begin + threadIdx.x; pos < end; pos += THREADS_PER_BLOCK_CSR){
				const int colA = indices[pos];
				sum += A[pos] * B[colA * ldb1 + offset_b_part ];
			}
			sums[threadIdx.x] = sum;
			__syncthreads();

			/* hardcoded reduction for 32 threads */
			sums[threadIdx.x] += sums[threadIdx.x + 16];
			sums[threadIdx.x] += sums[threadIdx.x +  8];
			sums[threadIdx.x] += sums[threadIdx.x +  4];
			sums[threadIdx.x] += sums[threadIdx.x +  2];
			sums[threadIdx.x] += sums[threadIdx.x +  1];

			if(threadIdx.x == 0){
				const int offsetC =  offset_c_part  + colC * ldc2;
				C[offsetC] = beta * C[offsetC] + alpha * sums[0];
			}
		}
	}
}

template <typename Dtype>
__global__  void caffe_gpu_csr_gemm_kernel(const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const Dtype alpha, int nzz, const Dtype* A, const int* indices,const int* ptr, const Dtype* B, const Dtype beta,
		Dtype* C, const CBLAS_ORDER orderC){

	if (orderC == CblasRowMajor ){
		if (TransB == CblasNoTrans ){
			caffe_gpu_csr_gemm_kernel_core(M,  N,  K, alpha,nzz,  A, indices, ptr,  B, N,  1, beta, C,  N,1);
		}else{
			caffe_gpu_csr_gemm_kernel_core(M,  N,  K, alpha,nzz,  A, indices, ptr,  B, 1,  K, beta, C,  N,1);
		}
	}else{
		if (TransB == CblasNoTrans ){
			caffe_gpu_csr_gemm_kernel_core(M,  N,  K, alpha,nzz,  A, indices, ptr,  B, N,  1, beta, C,  1,M);
		}else{
			caffe_gpu_csr_gemm_kernel_core(M,  N,  K, alpha,nzz,  A, indices, ptr,  B, 1,  K, beta, C,  1,M);
		}
	}
}

template <typename Dtype>
__device__  void caffe_gpu_csr_rank1_update_kernel_core(const int M, const int N,
		const Dtype alpha, const Dtype* A, const int* indices,const int* ptr, const Dtype* B, int ldb,
		Dtype* C,const int ldc1, const int ldc2){

	const int begin = ptr[0];
	const int end = ptr[1];
	for( int pos = blockIdx.x*blockDim.x + begin + threadIdx.x; pos < end; pos += blockDim.x * gridDim.x ){
		const int rowC = indices[pos];
		const Dtype valA = A[pos] * alpha;
		const int offset_part = rowC * ldc1;
		for( int colC = blockIdx.y*blockDim.y + threadIdx.y;  colC < N; colC += blockDim.y * gridDim.y ){
			const int C_offset  = offset_part  + colC * ldc2;
			C[C_offset] += B[colC * ldb] * valA;
		}
	}
}

//C = alpha A * B^T +  C where A and B are vectors. A is a sprase vector and B is a dense vector
template <typename Dtype>
__device__  void caffe_gpu_csr_rank1_update_kernel(const int M, const int N,
		const Dtype alpha, const Dtype* A, const int* indices,const int* ptr, const Dtype* B, int ldb,
		Dtype* C,const CBLAS_ORDER orderC){
	if (orderC == CblasRowMajor ){
		caffe_gpu_csr_rank1_update_kernel_core(M,N, alpha, A, indices, ptr, B, ldb, C, N, 1);
	}else{
		caffe_gpu_csr_rank1_update_kernel_core(M,N, alpha, A, indices, ptr, B, ldb, C, 1, M);
	}
}

template <typename Dtype>
__global__  void caffe_gpu_csr_rank1_update_kernel_multi(const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const Dtype alpha, const Dtype* A, const int* indices,const int* ptr, const Dtype* B, int ldb,
		Dtype* C,const CBLAS_ORDER orderC){
	if (TransB == CblasNoTrans){
		for (int i=0; i < K; i++){
			caffe_gpu_csr_rank1_update_kernel( M,  N, alpha, A, indices, ptr + i, B+(N*i), 1,C,orderC);
		}
	}else{
		for (int i=0; i < K; i++){
			caffe_gpu_csr_rank1_update_kernel( M,  N, alpha, A, indices, ptr + i, B+i, K,C,orderC);
		}
	}
}

template <>
void caffe_gpu_csr_gemm<float>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, int nzz, const float* A, const int* indices, const int* ptr, const float* B, const float beta,
		float* C, const CBLAS_ORDER orderC) {
	if (TransA == CblasNoTrans){
		dim3    grids(M,N);
		dim3    threads(THREADS_PER_BLOCK_CSR, 1);
		caffe_gpu_csr_gemm_kernel<float><<<grids,threads>>>(TransB, M, N, K,alpha, nzz, A,  indices, ptr,B, beta, C, orderC);
	}else{
		//scale C by beta
		if (beta != 1.0){
			CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle() , M * N, &beta, C, 1));
		}
		dim3 grids(CAFFE_GET_2D_BLOCKS(nzz/K+1),CAFFE_GET_2D_BLOCKS(N));
		dim3 threads(CAFFE_GET_2D_THREADS(nzz/K+1), CAFFE_GET_2D_THREADS(N));
		caffe_gpu_csr_rank1_update_kernel_multi<float><<<grids,threads>>>(TransB, M,  N, K, alpha, A, indices, ptr , B, 1,C,orderC);
	}
}

template <>
void caffe_gpu_csr_gemm<double>(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const double alpha, int nzz, const double* A, const int* indices, const int* ptr, const double* B, const double beta,
		double* C, const CBLAS_ORDER orderC) {
	if (TransA == CblasNoTrans){
		dim3    grids(M,N);
		dim3    threads(THREADS_PER_BLOCK_CSR, 1);
		caffe_gpu_csr_gemm_kernel<double><<<grids,threads>>>(TransB , M, N, K,alpha, nzz, A,  indices, ptr,B, beta, C, orderC);
	}else{
		//scale C by beta
		if (beta != 1.0){
			CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle() , M * N, &beta, C, 1));
		}
		dim3 grids(CAFFE_GET_2D_BLOCKS(nzz/K+1),CAFFE_GET_2D_BLOCKS(N));
		dim3 threads(CAFFE_GET_2D_THREADS(nzz/K+1), CAFFE_GET_2D_THREADS(N));
		caffe_gpu_csr_rank1_update_kernel_multi<double><<<grids,threads>>>(TransB, M,  N, K, alpha, A, indices, ptr , B, 1,C,orderC);
	}
}





}  // namespace caffe
