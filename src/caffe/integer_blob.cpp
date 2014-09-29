#include "caffe/integer_blob.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void IntegerBlob<Dtype>::Reshape(const int num, const int channels,
                                const int height, const int width) {

  Blob<Dtype>::Reshape(num, channels, height,width);
  if (this->count_ > 0) {
        indices_.reset(new SyncedMemory(this->num_ * this-> channels_ * sizeof(int)));
    } else {
      indices_.reset(reinterpret_cast<SyncedMemory*>(NULL));
    }
}

template<typename Dtype>
IntegerBlob<Dtype>::IntegerBlob(const int num, const int channels,
                                const int height, const int width) {
  Reshape(num, channels, height, width);
}

template<typename Dtype>
const int* IntegerBlob<Dtype>::cpu_indices() const {
  CHECK(indices_);
  return (const int*) indices_->cpu_data();
}

template<typename Dtype>
const int* IntegerBlob<Dtype>::gpu_indices() const {
  CHECK(indices_);
  return (const int*) indices_->gpu_data();
}

template<typename Dtype>
int* IntegerBlob<Dtype>::mutable_cpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_cpu_data());
}

template<typename Dtype>
int* IntegerBlob<Dtype>::mutable_gpu_indices() {
  CHECK(indices_);
  return reinterpret_cast<int*>(indices_->mutable_gpu_data());
}

template<typename Dtype>
void IntegerBlob<Dtype>::CopyFrom(const Blob<Dtype>& source, bool copy_diff,
                                 bool reshape) {
  LOG(FATAL)<< "CopyFrom is not supported";
}

template<typename Dtype>
void IntegerBlob<Dtype>::FromProto(const BlobProto& proto) {
  LOG(FATAL)<< "FromProto is not supported";
}

template<typename Dtype>
void IntegerBlob<Dtype>::ToProto(BlobProto* proto, bool write_diff) const {
  LOG(FATAL)<< "ToProto is not supported";
}
INSTANTIATE_CLASS(IntegerBlob);
}  // namespace caffe

