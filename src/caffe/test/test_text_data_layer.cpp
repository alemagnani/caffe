#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "leveldb/db.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/integer_blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class TextDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TextDataLayerTest()
      : backend_(TextDataParameter_DB_LEVELDB),
        blob_top_data_(new IntegerBlob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()),
        seed_(1701) {}
  virtual void SetUp() {
    filename_.reset(new string());
    MakeTempDir(filename_.get());
    *filename_ += "/db";
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  // Fill the LevelDB with data: if unique_pixels, each pixel is unique but
  // all images are the same; else each image is unique but all pixels within
  // an image are the same.
  void FillLevelDB() {
    backend_ = TextDataParameter_DB_LEVELDB;
    LOG(INFO)<< "Using temporary leveldb " << *filename_;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status = leveldb::DB::Open(options, filename_->c_str(),
                                               &db);
    CHECK(status.ok());
    for (int i = 0; i < 5; ++i) {
      TextDatum datum;
      datum.set_label(i);
      datum.set_n_index(100);
      datum.set_height_data(2);
      datum.set_size(10);

      for (int j = 0; j < 10; ++j) {
              datum.mutable_data()->Add(2 * j * i);
              datum.mutable_data()->Add(2 * j * i + 1);
              datum.mutable_indices()->Add(j * (i +1));
      }
      stringstream ss;
      ss << i;
      db->Put(leveldb::WriteOptions(), ss.str(), datum.SerializeAsString());
    }
    delete db;
  }

  // Fill the LMDB with data: unique_pixels has same meaning as in FillLevelDB.
  void FillLMDB() {
    backend_ = TextDataParameter_DB_LMDB;
    LOG(INFO)<< "Using temporary lmdb " << *filename_;
    CHECK_EQ(mkdir(filename_->c_str(), 0744), 0)<< "mkdir " << filename_
    << "failed";
    MDB_env * env;
    MDB_dbi dbi;
    MDB_val mdbkey, mdbdata;
    MDB_txn * txn;
    CHECK_EQ(mdb_env_create(&env), MDB_SUCCESS)<< "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(env, 1099511627776), MDB_SUCCESS)  // 1TB
<<    "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(env, filename_->c_str(), 0, 0664), MDB_SUCCESS)<< "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(env, NULL, 0, &txn), MDB_SUCCESS)<< "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(txn, NULL, 0, &dbi), MDB_SUCCESS)<< "mdb_open failed";

    for (int i = 0; i < 5; ++i) {
      TextDatum datum;
      datum.set_label(i);
      datum.set_n_index(100);
      datum.set_height_data(2);
      datum.set_size(10);

      for (int j = 0; j < 10; ++j) {
        datum.mutable_data()->Add(2 * j * i);
        datum.mutable_data()->Add(2 * j * i + 1);
        datum.mutable_indices()->Add(j * (i +1));
      }
      stringstream ss;
      ss << i;

      string value;
      datum.SerializeToString(&value);
      mdbdata.mv_size = value.size();
      mdbdata.mv_data = reinterpret_cast<void*>(&value[0]);
      string keystr = ss.str();
      mdbkey.mv_size = keystr.size();
      mdbkey.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(txn, dbi, &mdbkey, &mdbdata, 0), MDB_SUCCESS)<< "mdb_put failed";
    }
    CHECK_EQ(mdb_txn_commit(txn), MDB_SUCCESS)<< "mdb_txn_commit failed";
    mdb_close(env, dbi);
    mdb_env_close(env);
  }

  void TestRead() {
    LayerParameter param;
    TextDataParameter* data_param = param.mutable_text_data_param();
    data_param->set_batch_size(5);
    data_param->set_source(filename_->c_str());
    data_param->set_backend(backend_);

    TextDataLayer<Dtype> layer(param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_data_->num(), 5);
    EXPECT_EQ(blob_top_data_->channels(), 10);
    EXPECT_EQ(blob_top_data_->height(), 2);
    EXPECT_EQ(blob_top_data_->width(), 1);
    EXPECT_EQ(blob_top_label_->num(), 5);
    EXPECT_EQ(blob_top_label_->channels(), 1);
    EXPECT_EQ(blob_top_label_->height(), 1);
    EXPECT_EQ(blob_top_label_->width(), 1);

    for (int iter = 0; iter < 100; ++iter) {
      layer.Forward(blob_bottom_vec_, &blob_top_vec_);
      for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(i, blob_top_label_->cpu_data()[i]);
      }
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 10; ++j) {
           EXPECT_EQ((i+1) * j, blob_top_data_->cpu_indices()[j + i * 10])
               << "debug: iter " << iter << " i " << i << " j " << j;
        }
      }
      for (int i = 0; i < 5; ++i) {
              for (int j = 0; j < 10; ++j) {
                 EXPECT_EQ(2 * j * i, blob_top_data_->cpu_data()[2 * j + i * 20])
                     << "debug: iter " << iter << " i " << i << " j " << j;
                 EXPECT_EQ(2 * j * i + 1, blob_top_data_->cpu_data()[2* j + 1 + i * 20])
                                      << "debug: iter " << iter << " i " << i << " j " << j;
              }
            }
    }
  }

  virtual ~TextDataLayerTest() { delete blob_top_data_; delete blob_top_label_; }

  TextDataParameter_DB backend_;
  shared_ptr<string> filename_;
  IntegerBlob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int seed_;
};

TYPED_TEST_CASE(TextDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(TextDataLayerTest, TestReadLevelDB) {
  this->FillLevelDB();
  this->TestRead();
}

TYPED_TEST(TextDataLayerTest, TestReadLMDB) {
  this->FillLMDB();
  this->TestRead();
}


}  // namespace caffe
