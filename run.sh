#!/usr/bin/env bash

# export LD_LIBRARY_PATH="$HOME/git/scale/lib:$LD_LIBRARY_PATH"

cd ./build

source $HOME/git/scale/bin/scaleenv gfx1100
mangohud ./alien
