#ifndef FORMAT1D_H_
#define FORMAT1D_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

enum TensorFormat1D {
  // FORMAT_NWC is the default format in TensorFlow.
  FORMAT_NWC = 0,

  // FORMAT_NCW often improves performance on GPUs.
  FORMAT_NCW = 1,
};

bool Format1DFromString(absl::string_view format_str, TensorFormat1D* format) {
  if (format_str == "NWC") {
    *format = FORMAT_NWC;
    return true;
  }
  if (format_str == "NCW") {
    *format = FORMAT_NCW;
    return true;
  }
  return false;
}


inline int32 GetTensor1DDimIndex(TensorFormat1D format, char dimension) {
  if (format == FORMAT_NWC ) {
    // clang-format off
    switch (dimension) {
      case 'N': return 0;
      case 'W': return 1;
      case 'C': return 2;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else if (format == FORMAT_NCW) {
    switch (dimension) {
      case 'N': return 0;
      case 'C': return 1;
      case 'W': return 2;
      default:
        LOG(FATAL) << "Invalid dimension: " << dimension;
        return -1;  // Avoid compiler warning about missing return value
    }
  } else {
    LOG(FATAL) << "Invalid format: " << static_cast<int>(format);
    return -1;  // Avoid compiler warning about missing return value
  }
  // clang-format on
}

std::vector<int> get_tensor_shape(const Tensor& tensor)
{
    std::vector<int> shape;
    int num_dimensions = tensor.shape().dims();
    for(int ii_dim=0; ii_dim<num_dimensions; ii_dim++) {
        shape.push_back(tensor.shape().dim_size(ii_dim));
    }
    return shape;
}


template <typename T>
T GetTensor1DDim(std::vector<T> dimension_attributes, TensorFormat1D tensor_format, char dimension) {
  int index = GetTensor1DDimIndex(tensor_format, dimension);
  return dimension_attributes[index];
}

inline int GetTensor1DDim(const Tensor& tensor, TensorFormat1D tensor_format, char dimension) {
  return GetTensor1DDim(get_tensor_shape(tensor), tensor_format, dimension);
}


inline TensorShape ShapeFromFormat1D(TensorFormat1D format, int64_t N, int64_t W, int64_t C) {
  if (format == FORMAT_NWC ) {
      return TensorShape({N,W,C});
  }
  else {
      return TensorShape({N,C,W});
  }
}

}

#endif  // FORMAT1D_H_