
hip_add_library(EngineGpuKernels STATIC
    ConstantMemory.hip
    DataAccessKernels.hip
    DataAccessKernelsLauncher.hip
    DebugKernels.hip
    Definitions.h
    EditKernels.hip
    EditKernelsLauncher.hip
    FlowFieldKernels.hip
    GarbageCollectorKernels.hip
    GarbageCollectorKernelsLauncher.hip
    MaxAgeBalancer.hip
    Objects.hip
    RenderingData.hip
    RenderingKernels.hip
    RenderingKernelsLauncher.hip
    SimulationCudaFacade.hip
    SimulationData.hip
    SimulationKernels.hip
    SimulationKernelsLauncher.hip
    StatisticsService.hip
    StatisticsKernelsLauncher.hip
    StatisticsKernels.hip
    TestKernels.hip
    TestKernelsLauncher.hip
    )

target_link_libraries(EngineGpuKernels Base)
target_link_libraries(EngineGpuKernels EngineInterface)

# Link with HIP libraries - use the correct target name for ROCm/HIP
target_link_libraries(EngineGpuKernels ${HIP_LIBRARIES})

set_property(TARGET EngineGpuKernels PROPERTY HIP_SEPARABLE_COMPILATION ON)
set_property(TARGET EngineGpuKernels PROPERTY HIP_RESOLVE_DEVICE_SYMBOLS ON)