#!/bin/bash

set -exo pipefail

# gfx1100 == RX 7900XTX
# source $HOME/git/scale/bin/scaleenv gfx1100

export PATH="../cmake-3.3/bin:$PATH"

# Load ROCM HIP compiler and library
export ROCM_PATH=$(hipconfig --path)
export HIP_PLATFORM=amd
export HIP_PATH=$ROCM_PATH           # some tools still look for it
export HIP_CLANG_PATH=$ROCM_PATH/llvm/bin
export HIP_INCLUDE_PATH=$ROCM_PATH/include
export HIP_LIB_PATH=$ROCM_PATH/lib
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/amdgcn/bitcode   # device bitcode libs

# search paths
export PATH=$ROCM_PATH/bin:$HIP_CLANG_PATH:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH
export CPATH=$HIP_INCLUDE_PATH:$CPATH           # for clang/gcc
export PKG_CONFIG_PATH=$ROCM_PATH/lib/pkgconfig:$PKG_CONFIG_PATH

cmake \
    -S . \
    -B build \
    -DCMAKE_MODULE_PATH=/opt/rocm/lib/cmake/hip \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES="gfx1100"

cmake --build build -j "$(nproc)"
