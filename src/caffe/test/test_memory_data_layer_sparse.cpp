#include <vector>

#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MemoryDataLayerSparseTest : public ::testing::Test {
 protected:
  MemoryDataLayerSparseTest()
    : data_(new SparseBlob<Dtype>()),
      labels_(new Blob<Dtype>()),
      data_blob_(new SparseBlob<Dtype>()),
      label_blob_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    batch_size_ = 8;
    batches_ = 12;
    channels_ = 5;
    blob_top_vec_.push_back(data_blob_);
    blob_top_vec_.push_back(label_blob_);
    // pick random input data
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    data_->Reshape(batches_ * batch_size_, channels_, batches_ * batch_size_);
    labels_->Reshape(batches_ * batch_size_, 1, 1, 1);

    Dtype* d = data_->mutable_cpu_data();
    int*  ind = data_->mutable_cpu_indices();
    int*  ptr = data_->mutable_cpu_ptr();

    ptr[0] = 0;
    for (int k = 0; k < batches_ * batch_size_; k++) {
        ptr[k+1] = k+1;
        ind[k] = k % channels_;
        d[k] = (Dtype) k;
    }
    filler.Fill(this->labels_);
  }

  virtual ~MemoryDataLayerSparseTest() {
    delete data_blob_;
    delete label_blob_;
    delete data_;
    delete labels_;
  }
  int batch_size_;
  int batches_;
  int channels_;

  // we don't really need blobs for the input data, but it makes it
  //  easier to call Filler
  SparseBlob<Dtype>* const data_;

  Blob<Dtype>* const labels_;
  // blobs for the top of MemoryDataLayer
  SparseBlob<Dtype>* const data_blob_;
  Blob<Dtype>* const label_blob_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MemoryDataLayerSparseTest, TestDtypes);

TYPED_TEST(MemoryDataLayerSparseTest, TestSetup) {
  LayerParameter layer_param;
  MemoryDataSparseParameter* md_param = layer_param.mutable_memory_data_sparse_param();
  md_param->set_batch_size(this->batch_size_);
  md_param->set_size(this->channels_);

  shared_ptr<Layer<TypeParam> > layer(
      new MemoryDataLayerSparse<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->data_blob_->num(), this->batch_size_);
  EXPECT_EQ(this->data_blob_->channels(), this->channels_);
  EXPECT_EQ(this->data_blob_->height(), 1);
  EXPECT_EQ(this->data_blob_->width(), 1);
  EXPECT_EQ(this->label_blob_->num(), this->batch_size_);
  EXPECT_EQ(this->label_blob_->channels(), 1);
  EXPECT_EQ(this->label_blob_->height(), 1);
  EXPECT_EQ(this->label_blob_->width(), 1);
}

// run through a few batches and check that the right data appears
TYPED_TEST(MemoryDataLayerSparseTest, TestForward) {
  LayerParameter layer_param;
  MemoryDataSparseParameter* md_param = layer_param.mutable_memory_data_sparse_param();
  md_param->set_batch_size(this->batch_size_);
  md_param->set_size(this->channels_);
  shared_ptr<MemoryDataLayerSparse<TypeParam> > layer(
      new MemoryDataLayerSparse<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

  layer->Reset(this->data_->mutable_cpu_data(), this->data_->mutable_cpu_indices(),this->data_->mutable_cpu_ptr(),
      this->labels_->mutable_cpu_data(), this->data_->num() ,this->data_->channels());

  for (int i = 0; i < this->batches_ * 6; ++i) {
    int batch_num = i % this->batches_;
    int pos = batch_num * this->batch_size_;
    layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

    const TypeParam* d = this->data_blob_->cpu_data();
    const int* ind = this->data_blob_->cpu_indices();
    const int* ptr = this->data_blob_->cpu_ptr();
    EXPECT_EQ(this->data_blob_->num(), this->batch_size_);
    EXPECT_EQ(this->data_blob_->channels(), this->channels_);
    EXPECT_EQ(this->data_blob_->nzz(), this->batch_size_);
    EXPECT_EQ(ptr[this->batch_size_]- ptr[0], this->batch_size_);
    for (int j = 0; j < this->batch_size_; ++j) {
      EXPECT_EQ(d[ptr[j]], j + pos);
      EXPECT_EQ(ptr[j]+1, ptr[j+1]);
      EXPECT_EQ(ind[ptr[j]], (j +pos) % this->channels_);
    }
    for (int j = 0; j < this->label_blob_->count(); ++j) {
      EXPECT_EQ(this->label_blob_->cpu_data()[j],
          this->labels_->cpu_data()[this->batch_size_ * batch_num + j]);
    }
  }
}

}  // namespace caffe
