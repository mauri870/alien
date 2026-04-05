
add_library(EngineGpuKernels STATIC
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

# Link with HIP libraries
target_link_libraries(EngineGpuKernels ${HIP_LIBRARIES})

# zlib is required by HIP's RDC fat binary compression
target_link_libraries(EngineGpuKernels ZLIB::ZLIB)

set_property(TARGET EngineGpuKernels PROPERTY HIP_SEPARABLE_COMPILATION ON)
set_property(TARGET EngineGpuKernels PROPERTY HIP_RESOLVE_DEVICE_SYMBOLS ON)

# Enable relocatable device code for cross-TU device function calls.
# -fgpu-rdc must be used at both compile time and final link time.
# INTERFACE propagates the link flag to all executables consuming this library.
target_compile_options(EngineGpuKernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
target_link_options(EngineGpuKernels INTERFACE -fgpu-rdc)