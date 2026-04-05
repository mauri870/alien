#include "hip/hip_runtime.h"
#pragma once

#include "Math.hip.h"
#include "Map.hip.h"
#include "SimulationData.hip.h"

__global__ void cudaApplyFlowFieldSettings(SimulationData data);