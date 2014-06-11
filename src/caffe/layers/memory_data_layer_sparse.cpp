// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
     vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Memory Data Sparse Layer takes no blobs as input.";
  CHECK_EQ(top->size(), 2) << "Memory Data Layer Sparse takes two blobs as output.";
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  datum_size_ = this->layer_param_.memory_data_param().size();
  CHECK_GT(batch_size_ * datum_size_, 0) << "batch_size, channels, height,"
    " and width must be specified and positive in memory_data_param";
  (*top)[0]->Reshape(batch_size_, 1, 1, datum_size_);
  (*top)[1]->Reshape(batch_size_, 1, 1, 1);
  data_ = NULL;
  indices_ = NULL;
  ptr_ = NULL;
  labels_ = NULL;
}

template <typename Dtype>
void MemoryDataLayerSparse<Dtype>::Reset(Dtype* data, int* indices, int* ptr,  Dtype* label, int cols,  int rows) {
  CHECK(data);
  CHECK(labels);
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
  Dtype* toWrite =  (*top)[0]->mutable_cpu_data();

  memset(toWrite, 0, sizeof(Dtype) * batch_size_ * datum_size_);

  for (int r=0; r < batch_size_; r++){
	  const int begin = ptr_[pos+r];
  	  const int end = ptr_[pos+1+r];
  	  for (int p=begin; p < end; p++){
  		  toWrite[r * data_size_ + indices_[p]] = data_[p];
  	  }
  }
  (*top)[1]->set_cpu_data(labels_ + pos_);

  pos_ = (pos_ + batch_size_);
  if (pos_ >= rows_){ //notice that few data points will be lost if the rows are not nultiple of the batch size
	  pos__ = 0;
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(MemoryDataLayerSparse);

}  // namespace caffe
