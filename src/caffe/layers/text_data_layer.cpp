#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
TextDataLayer<Dtype>::~TextDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // clean up the database resources
  switch (this->layer_param_.text_data_param().backend()) {
  case TextDataParameter_DB_LEVELDB:
    break;  // do nothing
  case TextDataParameter_DB_LMDB:
    mdb_cursor_close(mdb_cursor_);
    mdb_close(mdb_env_, mdb_dbi_);
    mdb_txn_abort(mdb_txn_);
    mdb_env_close(mdb_env_);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
}

template <typename Dtype>
void TextDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

  // Initialize DB
  switch (this->layer_param_.text_data_param().backend()) {
  case TextDataParameter_DB_LEVELDB:
    {
    leveldb::DB* db_temp;
    leveldb::Options options = GetLevelDBOptions();
    options.create_if_missing = false;
    LOG(INFO) << "Opening leveldb " << this->layer_param_.text_data_param().source();
    leveldb::Status status = leveldb::DB::Open(
        options, this->layer_param_.text_data_param().source(), &db_temp);
    CHECK(status.ok()) << "Failed to open leveldb "
                       << this->layer_param_.text_data_param().source() << std::endl
                       << status.ToString();
    db_.reset(db_temp);
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();
    }
    break;
  case TextDataParameter_DB_LMDB:
    CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
    CHECK_EQ(mdb_env_open(mdb_env_,
             this->layer_param_.text_data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
    CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
    LOG(INFO) << "Opening lmdb " << this->layer_param_.text_data_param().source();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
        MDB_SUCCESS) << "mdb_cursor_get failed";
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
  // Read a data point, and use it to initialize the top blob.
  TextDatum datum;
  switch (this->layer_param_.text_data_param().backend()) {
  case TextDataParameter_DB_LEVELDB:
    datum.ParseFromString(iter_->value().ToString());
    break;
  case TextDataParameter_DB_LMDB:
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    break;
  default:
    LOG(FATAL) << "Unknown database backend";
  }
  height_data = datum.height_data();
  window_size = datum.size();

  this->prefetch_data_.reset(new IntegerBlob<Dtype>());
  this->prefetch_data_copy_.reset(new IntegerBlob<Dtype>());
  this->prefetch_label_.reset(new Blob<Dtype>());
  this->prefetch_label_copy_.reset(new Blob<Dtype>());

  (*top)[0]->Reshape(
        this->layer_param_.text_data_param().batch_size(),height_data, window_size, 1);
  (*top)[1]->Reshape(
          this->layer_param_.text_data_param().batch_size(),height_data, window_size, 1);

  this->prefetch_data_->Reshape(this->layer_param_.text_data_param().batch_size(), height_data, window_size, 1);
  this->prefetch_data_copy_->Reshape(this->layer_param_.text_data_param().batch_size(), height_data, window_size, 1);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (this->output_labels_) {
    (*top)[2]->Reshape(this->layer_param_.text_data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_->Reshape(this->layer_param_.text_data_param().batch_size(),
        1, 1, 1);
    this->prefetch_label_copy_->Reshape(this->layer_param_.text_data_param().batch_size(),
            1, 1, 1);
  }
}

template <typename Dtype>
void TextDataLayer<Dtype>::CopyData(Blob<Dtype>* top_blob){
  // Copy the data
  top_blob->ShareData(*this->prefetch_data_copy_.get());
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void TextDataLayer<Dtype>::InternalThreadEntry() {
  TextDatum datum;
  CHECK(this->prefetch_data_->count());

  Dtype* top_data = this->prefetch_data_->mutable_cpu_data();
  IntegerBlob<Dtype> * integerBlob =
          dynamic_cast<IntegerBlob<Dtype>*>(this->prefetch_data_.get());

  int* int_data = integerBlob->mutable_cpu_indices();

  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_) {
    top_label = this->prefetch_label_->mutable_cpu_data();
  }
  const int batch_size = this->layer_param_.text_data_param().batch_size();

  const int dtype_size = height_data * window_size;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    switch (this->layer_param_.text_data_param().backend()) {
    case TextDataParameter_DB_LEVELDB:
      CHECK(iter_);
      CHECK(iter_->Valid());
      datum.ParseFromString(iter_->value().ToString());
      break;
    case TextDataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
      datum.ParseFromArray(mdb_value_.mv_data,
          mdb_value_.mv_size);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    Dtype* destination = top_data + item_id * dtype_size;
    for (int k=0; k < dtype_size; k++) {
      destination[k] = datum.data(k);
    }
    int * int_destination = int_data + item_id * window_size;
    for (int k=0; k < window_size; k++) {
      int_destination[k] = datum.indices(k);
    }

    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }

    // go to the next iter
    switch (this->layer_param_.text_data_param().backend()) {
    case TextDataParameter_DB_LEVELDB:
      iter_->Next();
      if (!iter_->Valid()) {
        // We have reached the end. Restart from the first.
        // DLOG(INFO) << "Restarting data prefetching from start.";
        iter_->SeekToFirst();
      }
      break;
    case TextDataParameter_DB_LMDB:
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }
}

INSTANTIATE_CLASS(TextDataLayer);

}  // namespace caffe
