#!/bin/bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm -rf libtorch-shared-with-deps-latest.zip
git clone https://github.com/pytorch/vision.git
cd vision/
mkdir build
cd build
cmake ..
make
sudo make install
