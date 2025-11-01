
hip_add_library(EngineGpuKernels
    ActiveRadiationSources.cuh
    Array.cuh
    AttackerProcessor.cuh
    Base.cuh
    CellComputationProcessor.cuh
    CellConnectionProcessor.cuh
    CellFunctionProcessor.cuh
    CellProcessor.cuh
    ClusterProcessor.cuh
    ConstantMemory.cu
    ConstantMemory.cuh
    ConstructorProcessor.cuh
    CudaMemoryManager.cuh
    CudaNumberGenerator.cuh
    CudaShapeGenerator.cuh
    DataAccessKernels.cu
    DataAccessKernels.cuh
    DataAccessKernelsLauncher.cu
    DataAccessKernelsLauncher.cuh
    DebugKernels.cu
    DebugKernels.cuh
    Definitions.cuh
    Definitions.h
    DensityMap.cuh
    DetonatorProcessor.cuh
    EditKernels.cu
    EditKernels.cuh
    EditKernelsLauncher.cu
    EditKernelsLauncher.cuh
    FlowFieldKernels.cu
    FlowFieldKernels.cuh
    GarbageCollectorKernels.cu
    GarbageCollectorKernels.cuh
    GarbageCollectorKernelsLauncher.cu
    GarbageCollectorKernelsLauncher.cuh
    GenomeDecoder.cuh
    HashMap.cuh
    HashSet.cuh
    InjectorProcessor.cuh
    List.cuh
    Macros.cuh
    Map.cuh
    MapSectionCollector.cuh
    Math.cuh
    MaxAgeBalancer.cu
    MaxAgeBalancer.cuh
    MuscleProcessor.cuh
    MutationProcessor.cuh
    NerveProcessor.cuh
    NeuronProcessor.cuh
    Object.cuh
    Objects.cu
    Objects.cuh
    ObjectFactory.cuh
    Operations.cuh
    ParticleProcessor.cuh
    Physics.cuh
    PreprocessedSimulationData.cuh
    ReconnectorProcessor.cuh
    RenderingData.cu
    RenderingData.cuh
    RenderingKernels.cu
    RenderingKernels.cuh
    RenderingKernelsLauncher.cu
    RenderingKernelsLauncher.cuh
    SelectionResult.cuh
    SensorProcessor.cuh
    SimulationCudaFacade.cu
    SimulationCudaFacade.cuh
    SimulationData.cu
    SimulationData.cuh
    SimulationKernels.cu
    SimulationKernels.cuh
    SimulationKernelsLauncher.cu
    SimulationKernelsLauncher.cuh
    SimulationStatistics.cuh
    SpotCalculator.cuh
    StatisticsService.cu
    StatisticsService.cuh
    StatisticsKernelsLauncher.cu
    StatisticsKernelsLauncher.cuh
    StatisticsKernels.cu
    StatisticsKernels.cuh
    TestKernels.cu
    TestKernels.cuh
    TestKernelsLauncher.cu
    TestKernelsLauncher.cuh
    TransmitterProcessor.cuh
    TOs.cuh
    Util.cuh
    )

target_link_libraries(EngineGpuKernels Base)
target_link_libraries(EngineGpuKernels EngineInterface)

# Link with HIP libraries - use the correct target name for ROCm/HIP
target_link_libraries(EngineGpuKernels ${HIP_LIBRARIES})

set_property(TARGET EngineGpuKernels PROPERTY HIP_SEPARABLE_COMPILATION ON)
set_property(TARGET EngineGpuKernels PROPERTY HIP_RESOLVE_DEVICE_SYMBOLS ON)