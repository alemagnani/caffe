// Copyright 2014 BVLC and contributors.

#include "caffe/caffe.hpp"

int main(int argc, char** argv) {
  LOG(ERROR) << "Deprecated. Use caffe.bin train --solver_proto_file=... "
                "[resume_point_file=...] instead.";
  return 0;
}
