// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/sparse_blob.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	//LOG(INFO) << "setting up the memory data layer sparse";
	CHECK_EQ(bottom.size(), 0) << "Memory Data Sparse Layer takes no blobs as input.";
	CHECK_EQ(top->size(), 2) << "Memory Data Layer Sparse takes two blobs as output.";
	batch_size_ = this->layer_param_.memory_data_sparse_param().batch_size();
	datum_size_ = this->layer_param_.memory_data_sparse_param().size();
	CHECK_GT(batch_size_ , 0) << "batch_size must be specified and positive in _sparse_param";
	(*top)[0]->Reshape(batch_size_, datum_size_, 1, 1 );
	(*top)[1]->Reshape(batch_size_, 1, 1, 1);
}

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::Reset(Dtype* data, int* indices, int* ptr,  Dtype* labels, int rows,  int cols) {
	CHECK(data);
	CHECK(labels);
	CHECK(indices);
	CHECK(ptr);
	CHECK_EQ(cols, datum_size_);
	CHECK(rows > batch_size_) << "rows must be more than the batch size";
	const int nzz = ptr[rows] - ptr[0];

	blob_.reset(new SparseBlob<Dtype>());
	blob_->Reshape(rows, cols, nzz);
	blob_->set_cpu_data(data, indices, ptr, nzz);

	labels_.reset(new SyncedMemory(rows * sizeof(Dtype)));
	labels_->set_cpu_data(labels);

	rows_ = rows;
	pos_ = 0;
}

template <typename Dtype>
 Dtype* MemoryDataLayerSparse<Dtype>::cpu_labels() const{
	CHECK(labels_);
	return ( Dtype*)labels_->cpu_data();
}

template <typename Dtype>
Dtype* MemoryDataLayerSparse<Dtype>::gpu_labels() const{
	CHECK(labels_);
	return ( Dtype*)labels_->gpu_data();
}

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK(blob_->cpu_data()) << "MemoryDataLayerSparse needs to be initialized by calling Reset";

	const int* ptr = blob_->cpu_ptr();
	if ( SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>( (*top)[0] )){
		const int nzz = ptr[pos_+batch_size_] - ptr[pos_];
		sparseBlob->set_cpu_data( const_cast<Dtype*>(blob_->cpu_data()),const_cast<int*>(blob_->cpu_indices()), const_cast<int*>(ptr)+pos_,nzz,blob_->nzz());  //this is a hack we should handle it differently
	}else{
		Dtype* toWrite =  (*top)[0]->mutable_cpu_data();
		memset(toWrite, 0, sizeof(Dtype) * batch_size_ * datum_size_);

		for (int r=0; r < batch_size_; r++){
			const int begin = ptr[pos_+r];
			const int end = ptr[pos_+1+r];
			for (int p=begin; p < end; p++){
				toWrite[r * datum_size_ + blob_->cpu_indices()[p]] = blob_->cpu_data()[p];
			}
		}
	}
	(*top)[1]->set_cpu_data(cpu_labels()+pos_);

	pos_ = (pos_ + batch_size_);
	if (pos_ > (rows_ - batch_size_)){ //notice that few data points will be lost if the rows are not nultiple of the batch size
		pos_ = 0;
	}
}
#ifdef CPU_ONLY
STUB_GPU_FORWARD(MemoryDataLayerSparse, Forward);
#endif
INSTANTIATE_CLASS(MemoryDataLayerSparse);
}  // namespace caffe
