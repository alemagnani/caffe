// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/sparse_blob.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 0) << "Memory Data Sparse Layer takes no blobs as input.";
	CHECK_EQ(top->size(), 2) << "Memory Data Layer Sparse takes two blobs as output.";
	batch_size_ = this->layer_param_.memory_data_sparse_param().batch_size();
	datum_size_ = this->layer_param_.memory_data_sparse_param().size();
	CHECK_GT(batch_size_ , 0) << "batch_size must be specified and positive in _sparse_param";
	(*top)[0]->Reshape(batch_size_, datum_size_, 1, 1 );
	(*top)[1]->Reshape(batch_size_, 1, 1, 1);
	data_ = NULL;
	indices_ = NULL;
	ptr_ = NULL;
	labels_ = NULL;
}

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::Reset(Dtype* data, int* indices, int* ptr,  Dtype* labels, int rows,  int cols) {
	CHECK(data);
	CHECK(labels);
	CHECK(indices);
	CHECK(ptr);
	CHECK_EQ(cols, datum_size_);
	CHECK(rows > batch_size_) << "rows must be more than the batch size";
	data_ = data;
	labels_ = labels;
	indices_ = indices;
	ptr_ = ptr;
	rows_ = rows;
	pos_ = 0;

}

template <typename Dtype>
Dtype MemoryDataLayerSparse<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK(data_) << "MemoryDataLayerSparse needs to be initialized by calling Reset";

	if ( SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>( (*top)[0] )){
		//LOG(INFO) << "writing to a sparse blob";
		const int begin = ptr_[pos_];
		const int end = ptr_[pos_+batch_size_];
		const int nzz = end - begin;
		sparseBlob->Reshape(batch_size_, datum_size_, nzz);

		Dtype* data = sparseBlob->mutable_cpu_data();
		int*  indices = sparseBlob->mutable_cpu_indices();
		int*  ptr = sparseBlob->mutable_cpu_ptr();
		for (int i=0; i < sparseBlob->nzz(); i++){
			data[i] = this->data_[i+begin];
			indices[i] = indices_[i+begin];
		}
		for(int i=0; i <= batch_size_; i++){
			ptr[i] = ptr_[i+pos_] - begin;
		}
	}else{
		Dtype* toWrite =  (*top)[0]->mutable_cpu_data();
		memset(toWrite, 0, sizeof(Dtype) * batch_size_ * datum_size_);

		for (int r=0; r < batch_size_; r++){
			const int begin = ptr_[pos_+r];
			const int end = ptr_[pos_+1+r];
			for (int p=begin; p < end; p++){
				toWrite[r * datum_size_ + indices_[p]] = data_[p];
			}
		}
	}
	(*top)[1]->set_cpu_data(labels_ + pos_);

	pos_ = (pos_ + batch_size_);
	//LOG(INFO) << "pos" << pos_;
	if (pos_ >= (rows_ - batch_size_)){ //notice that few data points will be lost if the rows are not nultiple of the batch size
		//LOG(INFO) << "pos: " << pos_ << ", rows: " << rows_;
		pos_ = 0;
	}
	return Dtype(0.);
}

INSTANTIATE_CLASS(MemoryDataLayerSparse);

}  // namespace caffe
