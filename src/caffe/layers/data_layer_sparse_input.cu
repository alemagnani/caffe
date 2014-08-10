// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
Dtype DataLayerSparseInput<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	// First, join the thread
	JoinPrefetchThread();
	shared_ptr<SparseBlob<Dtype> >  tmp = prefetch_data_;
	prefetch_data_ = prefetch_data_copy;
	prefetch_data_copy = tmp;
	// Start a new prefetch thread
	CreatePrefetchThread();

	if ( SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>( (*top)[0] )){
		sparseBlob->set_gpu_data( const_cast<Dtype*>(prefetch_data_copy->gpu_data()),const_cast<int*>(prefetch_data_copy->gpu_indices()), const_cast<int*>(prefetch_data_copy->gpu_ptr()),prefetch_data_copy->nzz(),prefetch_data_copy->nzz());
	}else{
		LOG(FATAL) << "The top blob in the data layer sparse is not sparse\n";
	}
	if (output_labels_) {
	    CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
	        prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
	        cudaMemcpyHostToDevice));
	  }
	return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayerSparseInput);

}  // namespace caffe
