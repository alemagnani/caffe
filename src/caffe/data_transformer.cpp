#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/core/core.hpp>

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param)
    : param_(param) {
  phase_ = Caffe::phase();
    // check if we want to have mean
  if (param_.has_mean_file()) {
    const string& mean_file = param.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand() % 2;
  const bool has_mean_file = param_.has_mean_file();
  const bool has_unit8 = data.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }

  int h_off = 0;
  int w_off = 0;
  Dtype datum_element;
  if (crop_size) {
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(height - crop_size + 1);
      w_off = Rand(width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }

    int top_index, data_index;
    for (int c = 0; c < datum_channels; ++c) {
      int top_index_c = c * crop_size;
      int data_index_c = c * datum_height + h_off;
      for (int h = 0; h < crop_size; ++h) {
        int top_index_h = (top_index_c + h) * crop_size;
        int data_index_h = (data_index_c + h) * datum_width + w_off;
        for (int w = 0; w < crop_size; ++w) {
          data_index = data_index_h + w;
          if (do_mirror) {
            top_index = top_index_h + (crop_size - 1 - w);
          } else {
            top_index = top_index_h + w;
          }
          if (has_unit8) {
            datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
          } else {
            datum_element = datum.float_data(data_index);
          }
          if (has_mean_file) {
            transformed_data[top_index] =
              (datum_element - mean[data_index]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  } else {
    for (int j = 0; j < size; ++j) {
      if (has_unit8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
      } else {
        datum_element = datum.float_data(j);
      }
      if (has_mean_file) {
        transformed_data[j] =
          (datum_element - mean[j]) * scale;
      } else {
        transformed_data[j] = datum_element * scale;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
   
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_EQ(datum_channels, channels);
  CHECK_GE(datum_height, height);
  CHECK_GE(datum_width, width);

  CHECK_EQ(transformed_blob->num(), 1) <<
    "transformed_blob should have num() = 1";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand() % 2;
  const bool has_mean_file = param_.has_mean_file();
  const bool has_unit8 = data.size()>0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (datum_height - crop_size);
      w_off = Rand() % (datum_width - crop_size);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  
  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < channels; ++c) {
    int top_index_c = c * height;
    int data_index_c = c * datum_height + h_off;
    for (int h = 0; h < height; ++h) {
      int top_index_h = (top_index_c + h) * width;
      int data_index_h = (data_index_c + h) * datum_width + w_off;
      for (int w = 0; w < width; ++w) {
        data_index = data_index_h + w;
        if (do_mirror) {
          top_index = top_index_h + (width - 1 - w);
        } else {
          top_index = top_index_h + w;
        }
        if (has_unit8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          transformed_data[top_index] = datum_element * scale;
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be smaller than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {

  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_EQ(img_channels, channels);
  CHECK_GE(img_height, height);
  CHECK_GE(img_width, width);

  CHECK_EQ(transformed_blob->num(), 1) <<
    "transformed_blob should have num() = 1";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand() % 2;
  const bool has_mean_file = param_.has_mean_file();
  
  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (img_height - crop_size);
      w_off = Rand() % (img_width - crop_size);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Dtype pixel;
  int top_index;
  for (int c = 0; c < channels; ++c) {
    int top_index_c = c * height;
    int mean_index_c = c * img_height + h_off;
    for (int h = 0; h < height; ++h) {
      int top_index_h = (top_index_c + h) * width;
      int mean_index_h = (mean_index_c + h) * img_width + w_off;
      for (int w = 0; w < width; ++w) {
        if (do_mirror) {
          top_index = top_index_h + (width - 1 - w);
        } else {
          top_index = top_index_h + w;
        }
        pixel = static_cast<Dtype>(
              cv_img.at<cv::Vec3b>(h + h_off, w + w_off)[c]);
        if (has_mean_file) {
          int mean_index = mean_index_h + w;
          transformed_data[top_index] = (pixel - mean[mean_index]) * scale;
        } else {
          transformed_data[top_index] = pixel * scale;
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {

  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();
 
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand() % 2;
  const bool has_mean_file = param_.has_mean_file();

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (input_height - crop_size);
      w_off = Rand() % (input_width - crop_size);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    } 
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale!=Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == Caffe::TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
