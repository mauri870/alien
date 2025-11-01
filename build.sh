#!/bin/bash

set -ETeuo pipefail

# gfx1100 == RX 7900XTX
source $HOME/git/scale/bin/scaleenv gfx1100

export PATH="../cmake-3.3/bin:$PATH"

cmake \
    -S . \
    -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS}" \
    -DCMAKE_CUDA_COMPILER="${CUDA_PATH}/bin/nvcc"

cmake --build build -j$(nproc)
