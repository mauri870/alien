#include "hip/hip_runtime.h"
#pragma once

#include "EngineInterface/Colors.h"
#include "EngineInterface/ZoomLevels.h"

#include "TOs.hip.h"
#include "Base.hip.h"
#include "GarbageCollectorKernels.hip.h"
#include "ObjectFactory.hip.h"
#include "Map.hip.h"
#include "SimulationData.hip.h"
#include "RenderingData.hip.h"

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

__global__ void cudaDrawBackground(uint64_t* imageData, int2 imageSize, int2 worldSize, float zoom, float2 rectUpperLeft, float2 rectLowerRight);
__global__ void cudaPrepareFilteringForRendering(Array<Cell*> filteredCells, Array<Particle*> filteredParticles);
__global__ void cudaFilterCellsForRendering(
    int2 worldSize,
    float2 rectUpperLeft,
    Array<Cell*> cells,
    Array<Cell*> filteredCells,
    int2 imageSize,
    float zoom);
__global__ void cudaFilterParticlesForRendering(
    int2 worldSize,
    float2 rectUpperLeft,
    Array<Particle*> particles,
    Array<Particle*> filteredParticles,
    int2 imageSize,
    float zoom);
__global__ void cudaDrawCells(
    uint64_t timestep,
    int2 worldSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Cell*> cells,
    uint64_t* imageData,
    int2 imageSize,
    float zoom);
__global__ void cudaDrawCellGlow(int2 worldSize, float2 rectUpperLeft, Array<Cell*> cells, uint64_t* imageData, int2 imageSize, float zoom);

__global__ void cudaDrawParticles(int2 worldSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Particle*> particles, uint64_t* imageData, int2 imageSize, float zoom);
__global__ void cudaDrawRadiationSources(uint64_t* targetImage, float2 rectUpperLeft, int2 worldSize, int2 imageSize, float zoom);
__global__ void cudaDrawRepetition(int2 worldSize, int2 imageSize, float2 rectUpperLeft, float2 rectLowerRight, uint64_t* imageData, float zoom);
