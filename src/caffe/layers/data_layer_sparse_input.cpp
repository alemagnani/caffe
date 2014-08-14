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
	vector<SparseDatum> datums;
	CHECK(layer->prefetch_data_);

	Dtype* top_label;
	if (layer->output_labels_) {
		top_label = layer->prefetch_label_->mutable_cpu_data();
	}

	const int batch_size = layer->layer_param_.data_sparse_input_param().batch_size();

	const int size = layer->datum_size_;

	for (int item_id = 0; item_id < batch_size; ++item_id) {
		CHECK(layer->iter_);
		CHECK(layer->iter_->Valid());
		SparseDatum datum;
		datum.ParseFromString(layer->iter_->value().ToString());
		datums.push_back(datum);
		if (layer->output_labels_) {
			top_label[item_id] = datum.label();
		}
		// go to the next iter
		layer->iter_->Next();
		if (!layer->iter_->Valid()) {
			// We have reached the end. Restart from the first.
			//LOG(INFO) << "Restarting data prefetching from start.";
			layer->iter_->SeekToFirst();
		}
	}
	int nn = 0;
	for (int i=0; i < batch_size; i++){
		nn += datums[i].nn();
	}
	layer->prefetch_data_->Reshape(batch_size, size, nn);

	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
	int* indices = layer->prefetch_data_->mutable_cpu_indices();
	int* ptr = layer->prefetch_data_->mutable_cpu_ptr();

	ptr[0] = 0;
	int pos = 0;
	for (int i=0; i < batch_size; i++){
		SparseDatum d = datums[i];
		for( int k = 0; k < d.nn(); k++){
			top_data[k+pos] = d.data(k);
			indices[k+pos] = d.indices(k);
		}
		pos += d.nn();
		ptr[i+1] = pos;
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
	//LOG(INFO) << "Datum sample nn: " << datum.nn() << " size: " << datum.size()<< " datum: " << iter_->value().ToString();

	if ( SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>( (*top)[0] )){
		sparseBlob -> Reshape(this->layer_param_.data_sparse_input_param().batch_size(),datum.size(),1);
	}else{
		LOG(FATAL) << "The top blob in the data layer sparse is not sparse\n";
	}
	prefetch_data_.reset(new SparseBlob<Dtype>(this->layer_param_.data_sparse_input_param().batch_size(),datum.size(),1));
	prefetch_data_copy_.reset(new SparseBlob<Dtype>(this->layer_param_.data_sparse_input_param().batch_size(),datum.size(),1));

	LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
			<< (*top)[0]->channels() << "," << (*top)[0]->height() << ","
			<< (*top)[0]->width();
	// label
	if (output_labels_) {
		(*top)[1]->Reshape(this->layer_param_.data_sparse_input_param().batch_size(), 1, 1, 1);
		prefetch_label_.reset(
				new Blob<Dtype>(this->layer_param_.data_sparse_input_param().batch_size(), 1, 1, 1));
		prefetch_label_copy_.reset(
						new Blob<Dtype>(this->layer_param_.data_sparse_input_param().batch_size(), 1, 1, 1));
	}
	// datum size
	datum_size_ = datum.size();

	// Now, start the prefetch thread. Before calling prefetch, we make two
	// cpu_data calls so that the prefetch thread does not accidentally make
	// simultaneous cudaMalloc calls when the main thread is running. In some
	// GPUs this seems to cause failures if we do not so.
	prefetch_data_->mutable_cpu_data();
	if (output_labels_) {
		prefetch_label_->mutable_cpu_data();
	}
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
	//we swap the prefetch data
	prefetch_data_.swap(prefetch_data_copy_);
	prefetch_label_.swap(prefetch_label_copy_);

	// Start a new prefetch thread ahead of any memory transfer
	CreatePrefetchThread();

	if ( SparseBlob<Dtype> * sparseBlob = dynamic_cast<SparseBlob<Dtype>*>( (*top)[0] )){
		sparseBlob->set_cpu_data( const_cast<Dtype*>(prefetch_data_copy_->cpu_data()),const_cast<int*>(prefetch_data_copy_->cpu_indices()), const_cast<int*>(prefetch_data_copy_->cpu_ptr()),prefetch_data_copy_->nzz(),prefetch_data_copy_->nzz());
	}else{
		LOG(FATAL) << "The top blob in the data layer sparse is not sparse\n";
	}
	if (output_labels_) {
		caffe_copy(prefetch_label_copy_->count(), prefetch_label_copy_->cpu_data(),
				(*top)[1]->mutable_cpu_data());
	}
	return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayerSparseInput);

}  // namespace caffe
