#include "hip/hip_runtime.h"
﻿#pragma once

#include "Math.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

__global__ void cudaApplyFlowFieldSettings(SimulationData data);