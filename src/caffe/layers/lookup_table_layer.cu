#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/integer_blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LookupTableLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // TODO implement on GPU
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void LookupTableLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // TODO implement this on GPU
   Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(LookupTableLayer);

}  // namespace caffe
