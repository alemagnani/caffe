// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void* DataLayerSparseInputPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  DataLayerSparseInput<Dtype>* layer = static_cast<DataLayerSparseInput<Dtype>*>(layer_pointer);
  CHECK(layer);
  SparseDatum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }

  const int batch_size = layer->layer_param_.data_sparse_input_param().batch_size();

  // datum scales
  const int nn = layer->datum_nn_;
  const int size = layer->datum_size_;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(layer->iter_);
    CHECK(layer->iter_->Valid());
    datum.ParseFromString(layer->iter_->value().ToString());
    const string& data = datum.data();
    const int32 indices;

	// we will prefer to use data() first, and then try float_data()
	if (data.size()) {
		for (int j = 0; j < nn; ++j) {
		  Dtype datum_element =
			  static_cast<Dtype>(static_cast<uint8_t>(data[j]));
		  top_data[item_id * size + datum.indices(j)] = datum_element;
		}
	} else {
		for (int j = 0; j < size; ++j) {
		  top_data[item_id * size + datum.indices(j)] = datum.float_data(j);
		}
	}
    if (layer->output_labels_) {
      top_label[item_id] = datum.label();
    }
    // go to the next iter
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->iter_->SeekToFirst();
    }
  }
  return static_cast<void*>(NULL);
}

template <typename Dtype>
DataLayerSparseInput<Dtype>::~DataLayerSparseInput<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void DataLayerSparseInput<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GE(top->size(), 1) << "Data Layer takes at least one blob as output.";
  CHECK_LE(top->size(), 2) << "Data Layer takes at most two blobs as output.";
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_sparse_input_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.data_sparse_input_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.data_sparse_input_param().source() << std::endl
      << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_sparse_input_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_sparse_input_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  SparseDatum datum;
  datum.ParseFromString(iter_->value().ToString());


(*top)[0]->Reshape(
	this->layer_param_.data_sparse_input_param().batch_size(), 1,
   1, datum.size());
prefetch_data_.reset(new Blob<Dtype>(
	this->layer_param_.data_sparse_input_param().batch_size(), 1,
	1, datum.size()));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_sparse_input_param().batch_size(), 1, 1, 1);
    prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.data_sparse_input_param().batch_size(), 1, 1, 1));
  }
  // datum size
  datum_nn_ = datum.nn();
  datum_size_ = datum.size();

  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void DataLayerSparseInput<Dtype>::CreatePrefetchThread() {
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, DataLayerSparseInputPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void DataLayerSparseInput<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}


template <typename Dtype>
Dtype DataLayerSparseInput<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayerSparseInput);

}  // namespace caffe
