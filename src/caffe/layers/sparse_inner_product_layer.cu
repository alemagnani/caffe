#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sparse_blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template<typename Dtype>
void InnerProductLayer<Dtype>::Forward_sparse_gpu(
    const SparseBlob<Dtype>* bottomSparseBlob, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottomSparseBlob->gpu_data();
  const int* bottom_indices = bottomSparseBlob->gpu_indices();
  const int* bottom_ptr = bottomSparseBlob->gpu_ptr();
  const int nzz = bottomSparseBlob->nzz();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_csr_gemm<Dtype>(CblasNoTrans, CblasTrans, this->M_, this->N_,
                            this->K_, (Dtype) 1., nzz, bottom_data,
                            bottom_indices, bottom_ptr, weight, (Dtype) 0.,
                            top_data, CblasRowMajor);

  if (this->bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->N_, 1,
                          (Dtype) 1., this->bias_multiplier_.gpu_data(),
                          this->blobs_[1]->gpu_data(), (Dtype) 1., top_data);
  }
}

template<typename Dtype>
void InnerProductLayer<Dtype>::Backward_sparse_gpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    const SparseBlob<Dtype>* bottomSparseBlob) {

  // Gradient with respect to weight
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottomSparseBlob->gpu_data();
    const int* bottom_indices = bottomSparseBlob->gpu_indices();
    const int* bottom_ptr = bottomSparseBlob->gpu_ptr();
    const int nzz = bottomSparseBlob->nzz();
    caffe_gpu_csr_gemm<Dtype>(CblasTrans, CblasNoTrans, this->K_, this->N_,
                              this->M_, (Dtype) 1., nzz, bottom_data,
                              bottom_indices, bottom_ptr, top_diff, (Dtype) 0.,
                              this->blobs_[0]->mutable_gpu_diff(),
                              CblasColMajor);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, (Dtype) 1., top_diff,
                          this->bias_multiplier_.gpu_data(), (Dtype) 0.,
                          this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down) {
    // LOG(FATAL) << "propagate down is not supported by sparse inner product";
  }
}

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe