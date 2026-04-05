#!/bin/bash

source $HOME/git/scale/bin/scaleenv gfx1100
cd /home/mauri870/git/alien/build && gamemoderun ./cli -i resources/autosave.sim -t 500 2>&1
