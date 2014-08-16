// Copyright 2014 BVLC and contributors.


#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/sparse_blob.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK(blob_->cpu_data()) << "MemoryDataLayerSparse needs to be initialized by calling Reset";
	if ( SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>( (*top)[0] )){
		const int* ptr = blob_->cpu_ptr(); //this is correct CPU because it's easier to read out of it to compute nzz
		const int nzz = ptr[pos_+batch_size_] - ptr[pos_];
		sparseBlob->set_gpu_data( const_cast<Dtype*>(blob_->gpu_data()),const_cast<int*>(blob_->gpu_indices()), const_cast<int*>(blob_->gpu_ptr())+pos_,nzz,blob_->nzz());

	}else{
		LOG(FATAL) << "Forward_gpu to dense operation not supported\n";
	}
	(*top)[1]->set_gpu_data(gpu_labels()+pos_);

	pos_ = (pos_ + batch_size_);
	if (pos_ >= (rows_ - batch_size_)){ //notice that few data points will be lost if the rows are not nultiple of the batch size
		pos_ = 0;
	}
}
INSTANTIATE_CLASS(MemoryDataLayerSparse);
}  // namespace caffe
