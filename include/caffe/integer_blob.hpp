#ifndef CAFFE_INTEGER_BLOB_HPP_
#define CAFFE_INTEGER_BLOB_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

namespace caffe {

template<typename Dtype>
class IntegerBlob : public Blob<Dtype> {
 public:
  IntegerBlob()
      : Blob<Dtype>(),
        indices_() {
  }

  explicit IntegerBlob(const int num, const int channels, const int height,
                       const int width);

  virtual void Reshape(const int num, const int channels, const int height,
                       const int width);


  inline const shared_ptr<SyncedMemory>& indices() const {
    CHECK(indices_);
    return indices_;
  }

  const int* cpu_indices() const;
  const int* gpu_indices() const;

  int* mutable_cpu_indices();
  int* mutable_gpu_indices();

  virtual void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  virtual void FromProto(const BlobProto& proto);
  virtual void ToProto(BlobProto* proto, bool write_diff = false) const;

  virtual void ShareData(const Blob<Dtype>& other);
  virtual void ShareDiff(const Blob<Dtype>& other);

 protected:
  shared_ptr<SyncedMemory> indices_;

  DISABLE_COPY_AND_ASSIGN(IntegerBlob);
};  // class IntegerBlob

}  // namespace caffe

#endif  // CAFFE_INTEGER_BLOB_HPP_
