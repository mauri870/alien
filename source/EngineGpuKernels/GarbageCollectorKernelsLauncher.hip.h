#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Definitions.hip.h"
#include "Macros.hip.h"
#include "Base.hip.h"
#include "GarbageCollectorKernels.hip.h"

class _GarbageCollectorKernelsLauncher
{
public:
    _GarbageCollectorKernelsLauncher();
    ~_GarbageCollectorKernelsLauncher();

    void cleanupAfterTimestep(GpuSettings const& gpuSettings, SimulationData const& simulationData);
    void cleanupAfterDataManipulation(GpuSettings const& gpuSettings, SimulationData const& simulationData);
    void copyArrays(GpuSettings const& gpuSettings, SimulationData const& simulationData);
    void swapArrays(GpuSettings const& gpuSettings, SimulationData const& simulationData);

private:
    //gpu memory
    bool* _cudaBool;
};
