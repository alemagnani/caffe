// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "caffe/sparse_blob.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SparseBlob<Dtype>::Reshape(const int num, const int channels, const int nzz) {
  CHECK_GE(num, 0);
  CHECK_GE(channels, 0);
  CHECK_GE(nzz, 0);

  this->num_ = num;
  this->channels_ = channels;
  this->height_ = 1;
  this->width_ = 1;
  this->count_ = this->num_ * this->channels_;
  if (this->count_) {
	this->data_.reset(new SyncedMemory(nzz_ * sizeof(Dtype)));
    indices_.reset(new SyncedMemory(nzz_ * sizeof(int)));
    ptr_.reset(new SyncedMemory((this->num_ + 1) * sizeof(int)));
  } else {
	this->data_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    indices_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    ptr_.reset(reinterpret_cast<SyncedMemory*>(NULL));
  }
}

void Reshape(const int num, const int channels, const int height,
   const int width){
	LOG(FATAL) << "reshape without nzz is not supported";
}

template <typename Dtype>
void SparseBlob<Dtype>::ReshapeLike(const SparseBlob<Dtype>& other) {
  Reshape(other.num(), other.channels(), other.height(), other.width());
}

template <typename Dtype>
SparseBlob<Dtype>::SparseBlob(const int num, const int channels, const int nzz) {
  Reshape(num, channels, nzz);
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
	LOG(FATAL) << "set_cpu_data is not supported";
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
	LOG(FATAL) << "cpu_diff is not supported";
	return NULL;
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
	LOG(FATAL) << "gpu_diff is not supported";
		return NULL;
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
	LOG(FATAL) << "cpu_mutable_diff is not supported";
			return NULL;
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
	LOG(FATAL) << "cpu_mutable_diff is not supported";
	return NULL;
}



template <typename Dtype>
const int* SparseBlob<Dtype>::cpu_indices() const {
	CHECK(indices_);
	return (const int*)indices_->cpu_data();
}

template <typename Dtype>
const int* SparseBlob<Dtype>::cpu_ptr() const {
	CHECK(ptr_);
	return (const int*)ptr_->cpu_data();
}

template <typename Dtype>
const int* SparseBlob<Dtype>::gpu_indices() const {
	CHECK(indices_);
	return (const int*)indices_->gpu_data();
}

template <typename Dtype>
const int* SparseBlob<Dtype>::gpu_ptr() const {
	CHECK(ptr_);
	return (const int*)ptr_->gpu_data();
}

template <typename Dtype>
int* SparseBlob<Dtype>::mutable_cpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_cpu_data());
}

template <typename Dtype>
int* SparseBlob<Dtype>::mutable_cpu_ptr() {
  CHECK(ptr_);
  return reinterpret_cast<int*>(ptr_->mutable_cpu_data());
}

template <typename Dtype>
int* SparseBlob<Dtype>::mutable_gpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_gpu_data());
}

template <typename Dtype>
int* SparseBlob<Dtype>::mutable_gpu_ptr() {
  CHECK(ptr_);
  return reinterpret_cast<int*>(ptr_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
	LOG(FATAL) << "ShareData is not supported";
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
	LOG(FATAL) << "ShareDiff is not supported";
}

template <typename Dtype>
void Blob<Dtype>::Update() {
	LOG(FATAL) << "Update is not supported";
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
	LOG(FATAL) << "CopyFrom is not supported";
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto) {
	LOG(FATAL) << "FromProto is not supported";
}

template <typename Dtype>
void Blob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
	LOG(FATAL) << "ToProto is not supported";
}

INSTANTIATE_CLASS(SparseBlob);

}  // namespace caffe

