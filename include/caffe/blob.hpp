#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class Blob {
 public:
  Blob()
       : data_(), diff_(), num_(0), channels_(0), height_(0), width_(0),
       count_(0) {}
  explicit Blob(const int num, const int channels, const int height,
    const int width);
  virtual void Reshape(const int num, const int channels, const int height,
    const int width);
  void ReshapeLike(const Blob& other);
  inline int num() const { return num_; }
  inline int channels() const { return channels_; }
  inline int height() const { return height_; }
  inline int width() const { return width_; }
  inline int count() const { return count_; }
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
//    CHECK_GE(n, 0);
//    CHECK_LE(n, num_);
//    CHECK_GE(channels_, 0);
//    CHECK_LE(c, channels_);
//    CHECK_GE(height_, 0);
//    CHECK_LE(h, height_);
//    CHECK_GE(width_, 0);
//    CHECK_LE(w, width_);
    return ((n * channels_ + c) * height_ + h) * width_ + w;
  }
  // Copy from source. If copy_diff is false, we copy the data; if copy_diff
  // is true, we copy the diff.
  virtual void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  virtual inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_data() + offset(n, c, h, w));
  }

  virtual inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return *(cpu_diff() + offset(n, c, h, w));
  }

  virtual inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  virtual inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  virtual const Dtype* cpu_data() const;
  virtual void set_cpu_data(Dtype* data);
  virtual void set_gpu_data(Dtype* data);
  virtual const Dtype* gpu_data() const;
  virtual const Dtype* cpu_diff() const;
  virtual const Dtype* gpu_diff() const;
  virtual Dtype* mutable_cpu_data();
  virtual Dtype* mutable_gpu_data();
  virtual Dtype* mutable_cpu_diff();
  virtual Dtype* mutable_gpu_diff();
  virtual void Update();
  virtual void FromProto(const BlobProto& proto);
  virtual void ToProto(BlobProto* proto, bool write_diff = false) const;

  // Compute the sum of absolute values (L1 norm) of the data or diff.
  Dtype asum_data() const;
  Dtype asum_diff() const;

  // Set the data_/diff_ shared_ptr to point to the SyncedMemory holding the
  // data_/diff_ of Blob other -- useful in layers which simply perform a copy
  // in their forward or backward pass.
  // This deallocates the SyncedMemory holding this blob's data/diff, as
  // shared_ptr calls its destructor when reset with the = operator.
  virtual void ShareData(const Blob& other);
  virtual void ShareDiff(const Blob& other);

 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int count_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
