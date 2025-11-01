#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/Settings.h"

#include "Base.hip.h"
#include "DataAccessKernels.hip.h"
#include "Definitions.hip.h"
#include "GarbageCollectorKernelsLauncher.hip.h"
#include "Macros.hip.h"

class _RenderingKernelsLauncher
{
public:
    void drawImage(
        Settings const& settings,
        float2 rectUpperLeft,
        float2 rectLowerRight,
        int2 imageSize,
        float zoom,
        SimulationData data,
        RenderingData renderingData);
};
