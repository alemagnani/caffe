#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;

namespace caffe {

template <typename Dtype>
void TextDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

INSTANTIATE_LAYER_GPU_FUNCS(TextDataLayer);

}  // namespace caffe



