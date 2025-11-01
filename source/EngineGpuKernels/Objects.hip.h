#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.hip.h"
#include "Definitions.hip.h"
#include "Array.hip.h"

struct Objects
{
    Array<Cell*> cellPointers;
    Array<Particle*> particlePointers;

    Array<Cell> cells;
    Array<Particle> particles;

    RawMemory auxiliaryData;

    void init();
    void free();

    __device__ void saveNumEntries();
};

