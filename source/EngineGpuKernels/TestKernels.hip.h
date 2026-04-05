#include "hip/hip_runtime.h"
#pragma once

#include "hip/hip_runtime_api.h"

#include "EngineInterface/MutationType.h"
#include "SimulationData.hip.h"

__global__ void cudaTestMutate(SimulationData data, uint64_t cellId, MutationType mutationType);
