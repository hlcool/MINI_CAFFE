#!/usr/bin/env bash

./generatepb.sh

##### caffe without cuda and cudnn
mkdir -p build-caffe
pushd build-caffe
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
popd

##### caffe with cuda and cudnn
mkdir -p build-caffe-gpu
pushd build-caffe-gpu
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_CUDNN=ON ..
make -j4
popd



