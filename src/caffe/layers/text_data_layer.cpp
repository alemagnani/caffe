#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/dataset_factory.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
TextDataLayer<Dtype>::~TextDataLayer<Dtype>() {
  JoinPrefetchThread();
  // clean up the dataset resources
  dataset_->close();
}

template<typename Dtype>
void TextDataLayer<Dtype>::CreatePrefetchThread() {
  CHECK(StartInternalThread()) << "Thread execution failed";
}

template<typename Dtype>
void TextDataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
}

template<typename Dtype>
void TextDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {

  if (top.size() == MinTopBlobs()) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }

  // Initialize DB
  dataset_ = DatasetFactory<string, TextDatum>(
      this->layer_param_.text_data_param().backend());
  const string& source = this->layer_param_.text_data_param().source();
  LOG(INFO)<< "Opening dataset " << source;
  CHECK(dataset_->open(source, Dataset<string, TextDatum>::ReadOnly));
  iter_ = dataset_->begin();

  // Read a data point, and use it to initialize the top blob.
  CHECK(iter_ != dataset_->end());
  TextDatum datum = iter_->value;
  height_data = datum.height_data();
  window_size = datum.size();

  this->prefetch_data_.reset(new IntegerBlob<Dtype>());
  this->prefetch_data_copy_.reset(new IntegerBlob<Dtype>());
  this->prefetch_label_.reset(new Blob<Dtype>());
  this->prefetch_label_copy_.reset(new Blob<Dtype>());

  top[0]->Reshape(this->layer_param_.text_data_param().batch_size(),
                  height_data, window_size, 1);
  top[1]->Reshape(this->layer_param_.text_data_param().batch_size(),
                  height_data, window_size, 1);

  this->prefetch_data_->Reshape(
      this->layer_param_.text_data_param().batch_size(), height_data,
      window_size, 1);
  this->prefetch_data_copy_->Reshape(
      this->layer_param_.text_data_param().batch_size(), height_data,
      window_size, 1);
  LOG(INFO)<< "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();
  // label
  if (this->output_labels_) {
    top[2]->Reshape(this->layer_param_.text_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_->Reshape(
        this->layer_param_.text_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_copy_->Reshape(
        this->layer_param_.text_data_param().batch_size(), 1, 1, 1);
  }

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  DLOG(INFO)<< "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO)<< "Prefetch initialized.";

}

// This function is used to create a thread that prefetches the data.
template<typename Dtype>
void TextDataLayer<Dtype>::InternalThreadEntry() {
  CHECK(prefetch_data_->count());

  Dtype* top_data = prefetch_data_->mutable_cpu_data();
  int* int_data = prefetch_data_->mutable_cpu_indices();

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_->mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.text_data_param().batch_size();
  const int dtype_size = height_data * window_size;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    const TextDatum& datum = iter_->value;

    Dtype* destination = top_data + item_id * dtype_size;
    for (int k = 0; k < dtype_size; k++) {
      destination[k] = datum.data(k);
    }
    int * int_destination = int_data + item_id * window_size;
    for (int k = 0; k < window_size; k++) {
      int_destination[k] = datum.indices(k);
    }

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }

    ++iter_;
    if (iter_ == dataset_->end()) {
      iter_ = dataset_->begin();
    }
  }
}

template<typename Dtype>
void TextDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // we swap the prefetch data
  prefetch_data_.swap(prefetch_data_copy_);
  prefetch_label_.swap(prefetch_label_copy_);

  // Start a new prefetch thread ahead of any memory transfer
  CreatePrefetchThread();

  for (int k = 0; k < MinTopBlobs(); k++) {
    top[k]->ShareData(*prefetch_data_copy_.get());
  }

  if (output_labels_) {
    caffe_copy(prefetch_label_copy_->count(), prefetch_label_copy_->cpu_data(),
               top[MinTopBlobs()]->mutable_cpu_data());
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(TextDataLayer, Forward);
#endif

INSTANTIATE_CLASS(TextDataLayer);
REGISTER_LAYER_CLASS(TEXT_DATA, TextDataLayer);

}  // namespace caffe
