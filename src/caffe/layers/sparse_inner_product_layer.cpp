// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/sparse_blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
Dtype InnerProductLayer<Dtype>::Forward_sparse_cpu(const SparseBlob<Dtype>* bottomSparseBlob,
    vector<Blob<Dtype>*>* top) {

  const Dtype* bottom_data = bottomSparseBlob->cpu_data();
  const int*  bottom_indices = bottomSparseBlob->cpu_indices();
  const int* bottom_ptr = bottomSparseBlob->cpu_ptr();
  const int nzz = bottomSparseBlob->nzz();

  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_csr_gemm<Dtype>(CblasNoTrans, CblasTrans, this->M_, this->N_, this->K_, (Dtype)1., nzz,
      bottom_data, bottom_indices, bottom_ptr, weight, (Dtype)0., top_data, CblasRowMajor);

  if (this->bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(this->bias_multiplier_->cpu_data()),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_sparse_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    const SparseBlob<Dtype>*  bottomSparseBlob) {
  const Dtype* top_diff = top[0]->cpu_diff();

  const Dtype* bottom_data = bottomSparseBlob->cpu_data();
  const int*  bottom_indices = bottomSparseBlob->cpu_indices();
  const int* bottom_ptr = bottomSparseBlob->cpu_ptr();
  const int nzz = bottomSparseBlob->nzz();

  // Gradient with respect to weight
  caffe_cpu_csr_gemm<Dtype>(CblasTrans, CblasNoTrans, this->M_, this->K_, this->N_, (Dtype)1., nzz,
		  bottom_data, bottom_indices, bottom_ptr, top_diff, (Dtype)0., this->blobs_[0]->mutable_cpu_diff(),CblasColMajor);

  if (this->bias_term_) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, (Dtype)1., top_diff,
        reinterpret_cast<const Dtype*>(this->bias_multiplier_->cpu_data()), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down) {
	  LOG(FATAL) << "propagate down not supported for sparse inner product";
  }
}

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
