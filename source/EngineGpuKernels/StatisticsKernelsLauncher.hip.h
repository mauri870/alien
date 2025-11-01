#pragma once

#include "EngineInterface/GpuSettings.h"

#include "Base.hip.h"
#include "Definitions.hip.h"
#include "Macros.hip.h"

class _StatisticsKernelsLauncher
{
public:
    void updateStatistics(GpuSettings const& gpuSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics);

private:
};
