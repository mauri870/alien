#include "TestKernelsLauncher.hip.h"

#include "Macros.hip.h"
#include "TestKernels.hip.h"

void _TestKernelsLauncher::testOnly_mutate(GpuSettings const& gpuSettings, SimulationData const& data, uint64_t cellId, MutationType mutationType)
{
    KERNEL_CALL(cudaTestMutate, data, cellId, mutationType);
}
