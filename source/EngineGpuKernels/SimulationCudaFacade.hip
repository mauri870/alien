#include <glad/glad.h>

#include "SimulationCudaFacade.hip.h"

#include <functional>
#include <iostream>
#include <list>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "Base/Exceptions.h"
#include "Base/LoggingService.h"

#include "EngineInterface/InspectedEntityIds.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/SpaceCalculator.h"

#include "DataAccessKernels.hip.h"
#include "TOs.hip.h"
#include "Base.hip.h"
#include "GarbageCollectorKernels.hip.h"
#include "ConstantMemory.hip.h"
#include "CudaMemoryManager.hip.h"
#include "SimulationStatistics.hip.h"
#include "Objects.hip.h"
#include "Map.hip.h"
#include "StatisticsKernels.hip.h"
#include "EditKernels.hip.h"
#include "RenderingKernels.hip.h"
#include "SimulationData.hip.h"
#include "SimulationKernelsLauncher.hip.h"
#include "DataAccessKernelsLauncher.hip.h"
#include "RenderingKernelsLauncher.hip.h"
#include "EditKernelsLauncher.hip.h"
#include "StatisticsKernelsLauncher.hip.h"
#include "SelectionResult.hip.h"
#include "RenderingData.hip.h"
#include "TestKernelsLauncher.hip.h"
#include "StatisticsService.hip.h"

namespace
{
    std::chrono::milliseconds const StatisticsUpdate(30);
}

_SimulationCudaFacade::_SimulationCudaFacade(uint64_t timestep, Settings const& settings)
{
    initCuda();
    CudaMemoryManager::getInstance().reset();

    _settings.generalSettings = settings.generalSettings;
    setSimulationParameters(settings.simulationParameters);
    setGpuConstants(settings.gpuSettings);

    log(Priority::Important, "initialize simulation");

    _cudaSimulationData = std::make_shared<SimulationData>();
    _cudaRenderingData = std::make_shared<RenderingData>();
    _cudaSelectionResult = std::make_shared<SelectionResult>();
    _cudaAccessTO = std::make_shared<DataTO>();
    _cudaSimulationStatistics = std::make_shared<SimulationStatistics>();
    _statisticsService = std::make_shared<_StatisticsService>();

    _cudaSimulationData->init({settings.generalSettings.worldSizeX, settings.generalSettings.worldSizeY}, timestep);
    _cudaRenderingData->init();
    _cudaSimulationStatistics->init();
    _cudaSelectionResult->init();

    _simulationKernels = std::make_shared<_SimulationKernelsLauncher>();
    _dataAccessKernels = std::make_shared<_DataAccessKernelsLauncher>();
    _garbageCollectorKernels = std::make_shared<_GarbageCollectorKernelsLauncher>();
    _renderingKernels = std::make_shared<_RenderingKernelsLauncher>();
    _editKernels = std::make_shared<_EditKernelsLauncher>();
    _statisticsKernels = std::make_shared<_StatisticsKernelsLauncher>();

    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _cudaAccessTO->numAuxiliaryData);

    //default array sizes for empty simulation (will be resized later if not sufficient)
    resizeArrays({100000, 100000, 100000});
}

_SimulationCudaFacade::~_SimulationCudaFacade()
{
    _cudaSimulationData->free();
    _cudaRenderingData->free();
    _cudaSimulationStatistics->free();
    _cudaSelectionResult->free();

    _simulationKernels.reset();
    _dataAccessKernels.reset();
    _garbageCollectorKernels.reset();
    _renderingKernels.reset();
    _editKernels.reset();
    _statisticsKernels.reset();

    if (_cudaResource) {
        hipGraphicsUnregisterResource(_cudaResource);
        _cudaResource = nullptr;
    }
    if (_pbo) {
        glDeleteBuffers(1, &_pbo);
        _pbo = 0;
    }

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->auxiliaryData);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numAuxiliaryData);

    CHECK_FOR_CUDA_ERROR(hipDeviceReset());
    log(Priority::Important, "simulation closed");
}

void _SimulationCudaFacade::registerImageResource(GLuint textureId)
{
    _textureId = textureId;

    // Invalidate PBO so it gets recreated at the correct size in drawVectorGraphics.
    if (_cudaResource) {
        CHECK_FOR_CUDA_ERROR(hipGraphicsUnregisterResource(_cudaResource));
        _cudaResource = nullptr;
    }
    _pboSize = {0, 0};

    // Re-select the HIP device that is compatible with the current OpenGL context.
    // hipGLGetDevices must be called after an OpenGL context is current (guaranteed here
    // since we received a valid textureId). Without this, hipGraphicsGLRegisterBuffer fails.
    unsigned int glDeviceCount = 0;
    int glDevice = 0;
    auto glResult = hipGLGetDevices(&glDeviceCount, &glDevice, 1, hipGLDeviceListAll);
    if (glResult == hipSuccess && glDeviceCount > 0) {
        CHECK_FOR_CUDA_ERROR(hipSetDevice(glDevice));
        log(Priority::Important, "GL-interop HIP device: " + std::to_string(glDevice));
    }
}

void _SimulationCudaFacade::calcTimestep(uint64_t timesteps, bool forceUpdateStatistics)
{
    for (uint64_t i = 0; i < timesteps; ++i) {
        checkAndProcessSimulationParameterChanges();

        auto simulationData = getSimulationDataIntern();
        _simulationKernels->calcTimestep(_settings, simulationData, *_cudaSimulationStatistics);
        syncAndCheck();

        automaticResizeArrays();

        {
            std::lock_guard lock(_mutexForSimulationData);
            ++_cudaSimulationData->timestep;
        }
        auto statistics = getRawStatistics();
        {
            std::lock_guard lock(_mutexForSimulationParameters);
            if (_simulationKernels->updateSimulationParametersAfterTimestep(_settings, simulationData, statistics)) {
                CHECK_FOR_CUDA_ERROR(
                    hipMemcpyToSymbol(HIP_SYMBOL(cudaSimulationParameters), &_settings.simulationParameters, sizeof(SimulationParameters), 0, hipMemcpyHostToDevice));
            }
        }
        auto now = std::chrono::steady_clock::now();
        if (!_lastStatisticsUpdateTime || now - *_lastStatisticsUpdateTime > StatisticsUpdate) {
            updateStatistics();
            // Set timestamp AFTER completion so the interval is measured from end of last update,
            // preventing a feedback loop when updateStatistics() takes longer than StatisticsUpdate.
            _lastStatisticsUpdateTime = std::chrono::steady_clock::now();
        }
    }
    if (forceUpdateStatistics) {
        updateStatistics();
    }
}

void _SimulationCudaFacade::applyCataclysm(int power)
{
    for (int i = 0; i < power; ++i) {
        _editKernels->applyCataclysm(_settings.gpuSettings, getSimulationDataIntern());
        syncAndCheck();
        resizeArraysIfNecessary();
    }
}

void _SimulationCudaFacade::drawVectorGraphics(
    float2 const& rectUpperLeft,
    float2 const& rectLowerRight,
    int2 const& imageSize,
    double zoom)
{
    checkAndProcessSimulationParameterChanges();

    _cudaRenderingData->resizeImageIfNecessary(imageSize);

    _renderingKernels->drawImage(_settings, rectUpperLeft, rectLowerRight, imageSize, static_cast<float>(zoom), getSimulationDataIntern(), *_cudaRenderingData);
    syncAndCheck();

    // Create or resize the PBO if the image dimensions changed.
    if (_pboSize.x != imageSize.x || _pboSize.y != imageSize.y) {
        if (_cudaResource) {
            CHECK_FOR_CUDA_ERROR(hipGraphicsUnregisterResource(_cudaResource));
            _cudaResource = nullptr;
        }
        if (_pbo == 0) {
            glGenBuffers(1, &_pbo);
        }
        const size_t bufferSize = sizeof(uint64_t) * imageSize.x * imageSize.y;
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(bufferSize), nullptr, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CHECK_FOR_CUDA_ERROR(hipGraphicsGLRegisterBuffer(&_cudaResource, _pbo, hipGraphicsRegisterFlagsWriteDiscard));
        _pboSize = imageSize;
    }

    // Map PBO into HIP address space and copy rendered image data into it (device→device).
    CHECK_FOR_CUDA_ERROR(hipGraphicsMapResources(1, &_cudaResource));
    void* pboDevPtr = nullptr;
    size_t pboSize = 0;
    CHECK_FOR_CUDA_ERROR(hipGraphicsResourceGetMappedPointer(&pboDevPtr, &pboSize, _cudaResource));
    CHECK_FOR_CUDA_ERROR(hipMemcpy(pboDevPtr, _cudaRenderingData->imageDataDevice,
        sizeof(uint64_t) * imageSize.x * imageSize.y, hipMemcpyDeviceToDevice));
    CHECK_FOR_CUDA_ERROR(hipGraphicsUnmapResources(1, &_cudaResource));

    // Upload PBO to GL texture — the GL driver performs this as a GPU-side DMA transfer.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
    glBindTexture(GL_TEXTURE_2D, _textureId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageSize.x, imageSize.y, GL_RGBA, GL_UNSIGNED_SHORT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void _SimulationCudaFacade::getSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataTO const& dataTO)
{
    _dataAccessKernels->getData(_settings.gpuSettings, getSimulationDataIntern(), rectUpperLeft, rectLowerRight, *_cudaAccessTO);
    syncAndCheck();

    copyDataTOtoHost(dataTO);
}

void _SimulationCudaFacade::getSelectedSimulationData(bool includeClusters, DataTO const& dataTO)
{
    _dataAccessKernels->getSelectedData(_settings.gpuSettings, getSimulationDataIntern(), includeClusters, *_cudaAccessTO);
    syncAndCheck();

    copyDataTOtoHost(dataTO);
}

void _SimulationCudaFacade::getInspectedSimulationData(std::vector<uint64_t> entityIds, DataTO const& dataTO)
{
    InspectedEntityIds ids;
    if (entityIds.size() > Const::MaxInspectedObjects) {
        return;
    }
    for (int i = 0; i < entityIds.size(); ++i) {
        ids.values[i] = entityIds.at(i);
    }
    if (entityIds.size() < Const::MaxInspectedObjects) {
        ids.values[entityIds.size()] = 0;
    }
    _dataAccessKernels->getInspectedData(_settings.gpuSettings, getSimulationDataIntern(), ids, *_cudaAccessTO);
    syncAndCheck();
    copyDataTOtoHost(dataTO);
}

void _SimulationCudaFacade::getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataTO const& dataTO)
{
    _dataAccessKernels->getOverlayData(_settings.gpuSettings, getSimulationDataIntern(), rectUpperLeft, rectLowerRight, *_cudaAccessTO);
    syncAndCheck();

    copyToHost(dataTO.numCells, _cudaAccessTO->numCells);
    copyToHost(dataTO.numParticles, _cudaAccessTO->numParticles);
    copyToHost(dataTO.cells, _cudaAccessTO->cells, *dataTO.numCells);
    copyToHost(dataTO.particles, _cudaAccessTO->particles, *dataTO.numParticles);
}

void _SimulationCudaFacade::addAndSelectSimulationData(DataTO const& dataTO)
{
    copyDataTOtoDevice(dataTO);
    _editKernels->removeSelection(_settings.gpuSettings, getSimulationDataIntern());
    _dataAccessKernels->addData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaAccessTO, true, true);
    syncAndCheck();
    updateStatistics();
}

void _SimulationCudaFacade::setSimulationData(DataTO const& dataTO)
{
    copyDataTOtoDevice(dataTO);
    _dataAccessKernels->clearData(_settings.gpuSettings, getSimulationDataIntern());
    _dataAccessKernels->addData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaAccessTO, false, false);
    syncAndCheck();
    updateStatistics();
}

void _SimulationCudaFacade::removeSelectedObjects(bool includeClusters)
{
    _editKernels->removeSelectedObjects(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
    updateStatistics();
}

void _SimulationCudaFacade::relaxSelectedObjects(bool includeClusters)
{
    _editKernels->relaxSelectedObjects(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    _editKernels->uniformVelocities(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::makeSticky(bool includeClusters)
{
    _editKernels->makeSticky(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::removeStickiness(bool includeClusters)
{
    _editKernels->removeStickiness(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::setBarrier(bool value, bool includeClusters)
{
    _editKernels->setBarrier(_settings.gpuSettings, getSimulationDataIntern(), value, includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::changeInspectedSimulationData(DataTO const& changeDataTO)
{
    copyDataTOtoDevice(changeDataTO);
    _editKernels->changeSimulationData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaAccessTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();
}

void _SimulationCudaFacade::applyForce(ApplyForceData const& applyData)
{
    _editKernels->applyForce(_settings.gpuSettings, getSimulationDataIntern(), applyData);
    syncAndCheck();
}

void _SimulationCudaFacade::switchSelection(PointSelectionData const& pointData)
{
    _editKernels->switchSelection(_settings.gpuSettings, getSimulationDataIntern(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::swapSelection(PointSelectionData const& pointData)
{
    _editKernels->swapSelection(_settings.gpuSettings, getSimulationDataIntern(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::setSelection(AreaSelectionData const& selectionData)
{
    _editKernels->setSelection(_settings.gpuSettings, getSimulationDataIntern(), selectionData);
}

 SelectionShallowData _SimulationCudaFacade::getSelectionShallowData(float2 const& refPos)
{
    _editKernels->getSelectionShallowData(_settings.gpuSettings, getSimulationDataIntern(), refPos, * _cudaSelectionResult);
    syncAndCheck();
    return _cudaSelectionResult->getSelectionShallowData();
}

void _SimulationCudaFacade::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& shallowUpdateData)
{
    _editKernels->shallowUpdateSelectedObjects(_settings.gpuSettings, getSimulationDataIntern(), shallowUpdateData);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelection()
{
    _editKernels->removeSelection(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::updateSelection()
{
    _editKernels->updateSelection(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();
}

void _SimulationCudaFacade::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    _editKernels->colorSelectedCells(_settings.gpuSettings, getSimulationDataIntern(), color, includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::reconnectSelectedObjects()
{
    _editKernels->reconnect(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();
}

void _SimulationCudaFacade::setDetached(bool value)
{
    _editKernels->setDetached(_settings.gpuSettings, getSimulationDataIntern(), value);
    syncAndCheck();
}

void _SimulationCudaFacade::setGpuConstants(GpuSettings const& gpuConstants)
{
    _settings.gpuSettings = gpuConstants;

    CHECK_FOR_CUDA_ERROR(
        hipMemcpyToSymbol(HIP_SYMBOL(cudaThreadSettings), &gpuConstants, sizeof(GpuSettings), 0, hipMemcpyHostToDevice));
}

SimulationParameters _SimulationCudaFacade::getSimulationParameters() const
{
    std::lock_guard lock(_mutexForSimulationParameters);
    return _newSimulationParameters ? *_newSimulationParameters : _settings.simulationParameters;
}

void _SimulationCudaFacade::setSimulationParameters(SimulationParameters const& parameters)
{
    std::lock_guard lock(_mutexForSimulationParameters);
    _newSimulationParameters = parameters;
}

auto _SimulationCudaFacade::getArraySizes() const -> ArraySizes
{
    return {
        _cudaSimulationData->objects.cells.getSize_host(),
        _cudaSimulationData->objects.particles.getSize_host(),
        _cudaSimulationData->objects.auxiliaryData.getSize_host()
    };
}

RawStatisticsData _SimulationCudaFacade::getRawStatistics()
{
    std::lock_guard lock(_mutexForStatistics);
    if (_statisticsData) {
        return *_statisticsData;
    } else {
        return RawStatisticsData();
    }
}

void _SimulationCudaFacade::updateStatistics()
{
    _statisticsKernels->updateStatistics(_settings.gpuSettings, getSimulationDataIntern(), *_cudaSimulationStatistics);
    syncAndCheck();

    {
        std::lock_guard lock(_mutexForStatistics);
        _statisticsData = _cudaSimulationStatistics->getStatistics();
    }
    _statisticsService->addDataPoint(_statisticsHistory, _statisticsData->timeline, getCurrentTimestep());
}

StatisticsHistory const& _SimulationCudaFacade::getStatisticsHistory() const
{
    return _statisticsHistory;
}

void _SimulationCudaFacade::setStatisticsHistory(StatisticsHistoryData const& data)
{
    _statisticsService->rewriteHistory(_statisticsHistory, data, getCurrentTimestep());
}

void _SimulationCudaFacade::resetTimeIntervalStatistics()
{
    _cudaSimulationStatistics->resetAccumulatedStatistics();
}

uint64_t _SimulationCudaFacade::getCurrentTimestep() const
{
    std::lock_guard lock(_mutexForSimulationData);
    return _cudaSimulationData->timestep;
}

void _SimulationCudaFacade::setCurrentTimestep(uint64_t timestep)
{
    {
        std::lock_guard lock(_mutexForSimulationData);
        _cudaSimulationData->timestep = timestep;
    }
    _statisticsService->resetTime(_statisticsHistory, timestep);
}

void _SimulationCudaFacade::clear()
{
    _dataAccessKernels->clearData(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();
}

void _SimulationCudaFacade::resizeArraysIfNecessary(ArraySizes const& additionals)
{
    if (_cudaSimulationData->shouldResize(additionals)) {
        resizeArrays(additionals);
    }
}

void _SimulationCudaFacade::testOnly_mutate(uint64_t cellId, MutationType mutationType)
{
    {
        std::lock_guard lock(_mutexForSimulationParameters);
        if (_newSimulationParameters) {
            _settings.simulationParameters = *_newSimulationParameters;
            CHECK_FOR_CUDA_ERROR(
                hipMemcpyToSymbol(HIP_SYMBOL(cudaSimulationParameters), &*_newSimulationParameters, sizeof(SimulationParameters), 0, hipMemcpyHostToDevice));
            _newSimulationParameters.reset();
        }
    }
    _testKernels->testOnly_mutate(_settings.gpuSettings, getSimulationDataIntern(), cellId, mutationType);
    syncAndCheck();

    resizeArraysIfNecessary();
}

void _SimulationCudaFacade::initCuda()
{
    log(Priority::Important, "initialize CUDA");
    _gpuInfo = checkAndReturnGpuInfo();

    auto result = hipSetDevice(_gpuInfo.deviceNumber);
    if (result != hipSuccess) {
        throw SystemRequirementNotMetException("CUDA device could not be initialized.");
    }

    hipGetLastError(); //reset error code

    log(Priority::Important, "device " + std::to_string(_gpuInfo.deviceNumber) + " selected");
}

auto _SimulationCudaFacade::checkAndReturnGpuInfo() -> GpuInfo
{
    static std::optional<GpuInfo> cachedResult;
    if (cachedResult) {
        return *cachedResult;
    }
    cachedResult = GpuInfo();

    int numberOfDevices;
    CHECK_FOR_CUDA_ERROR(hipGetDeviceCount(&numberOfDevices));
    if (numberOfDevices < 1) {
        throw SystemRequirementNotMetException("No CUDA device found.");
    }
    {
        std::stringstream stream;
        if (1 == numberOfDevices) {
            stream << "1 CUDA device found";
        } else {
            stream << numberOfDevices << " CUDA devices found";
        }
        log(Priority::Important, stream.str());
    }

    int highestComputeCapability = 0;
    for (int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber) {
        hipDeviceProp_t prop;
        CHECK_FOR_CUDA_ERROR(hipGetDeviceProperties(&prop, deviceNumber));

        std::stringstream stream;
        stream << "device " << deviceNumber << ": " << prop.name << " with compute capability " << prop.major << "." << prop.minor;
        log(Priority::Important, stream.str());

        int computeCapability = prop.major * 100 + prop.minor;
        if (computeCapability > highestComputeCapability) {
            cachedResult->deviceNumber = deviceNumber;
            highestComputeCapability = computeCapability;
            cachedResult->gpuModelName = prop.name;
        }
    }
    if (highestComputeCapability < 600) {
        throw SystemRequirementNotMetException("No CUDA device with compute capability of 6.0 or higher found.");
    }

    return *cachedResult;
}

void _SimulationCudaFacade::syncAndCheck()
{
    hipDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(hipGetLastError());
}

void _SimulationCudaFacade::copyDataTOtoDevice(DataTO const& dataTO)
{
    copyToDevice(_cudaAccessTO->numCells, dataTO.numCells);
    copyToDevice(_cudaAccessTO->numParticles, dataTO.numParticles);
    copyToDevice(_cudaAccessTO->numAuxiliaryData, dataTO.numAuxiliaryData);

    copyToDevice(_cudaAccessTO->cells, dataTO.cells, *dataTO.numCells);
    copyToDevice(_cudaAccessTO->particles, dataTO.particles, *dataTO.numParticles);
    copyToDevice(_cudaAccessTO->auxiliaryData, dataTO.auxiliaryData, *dataTO.numAuxiliaryData);
}

void _SimulationCudaFacade::copyDataTOtoHost(DataTO const& dataTO)
{
    copyToHost(dataTO.numCells, _cudaAccessTO->numCells);
    copyToHost(dataTO.numParticles, _cudaAccessTO->numParticles);
    copyToHost(dataTO.numAuxiliaryData, _cudaAccessTO->numAuxiliaryData);

    copyToHost(dataTO.cells, _cudaAccessTO->cells, *dataTO.numCells);
    copyToHost(dataTO.particles, _cudaAccessTO->particles, *dataTO.numParticles);
    copyToHost(dataTO.auxiliaryData, _cudaAccessTO->auxiliaryData, *dataTO.numAuxiliaryData);
}

void _SimulationCudaFacade::automaticResizeArrays()
{
    uint64_t timestep;
    {
        std::lock_guard lock(_mutexForSimulationData);
        timestep = _cudaSimulationData->timestep;
    }
    //make check after every 10th time step
    if (timestep % 10 == 0) {
        resizeArraysIfNecessary();
    }
}

void _SimulationCudaFacade::resizeArrays(ArraySizes const& additionals)
{
    log(Priority::Important, "resize arrays");

    _cudaSimulationData->resizeTargetObjects(additionals);
    if (!_cudaSimulationData->isEmpty()) {
        _garbageCollectorKernels->copyArrays(_settings.gpuSettings, getSimulationDataIntern());
        syncAndCheck();

        _cudaSimulationData->resizeObjects();

        _garbageCollectorKernels->swapArrays(_settings.gpuSettings, getSimulationDataIntern());
        syncAndCheck();
    } else {
        _cudaSimulationData->resizeObjects();
    }

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->auxiliaryData);

    auto cellArraySize = _cudaSimulationData->objects.cells.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<CellTO>(cellArraySize, _cudaAccessTO->cells);
    auto particleArraySize = _cudaSimulationData->objects.particles.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<ParticleTO>(particleArraySize, _cudaAccessTO->particles);
    auto auxiliaryDataSize = _cudaSimulationData->objects.auxiliaryData.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<uint8_t>(auxiliaryDataSize, _cudaAccessTO->auxiliaryData);

    CHECK_FOR_CUDA_ERROR(hipGetLastError());

    log(Priority::Unimportant, "cell array size: " + std::to_string(cellArraySize));
    log(Priority::Unimportant, "particle array size: " + std::to_string(particleArraySize));
    log(Priority::Unimportant, "auxiliary data size: " + std::to_string(auxiliaryDataSize));

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    log(Priority::Important, std::to_string(memorySizeAfter / (1024 * 1024)) + " MB GPU memory used");
}

void _SimulationCudaFacade::checkAndProcessSimulationParameterChanges()
{
    std::lock_guard lock(_mutexForSimulationParameters);
    if (_newSimulationParameters) {
        _settings.simulationParameters = *_newSimulationParameters;
        CHECK_FOR_CUDA_ERROR(hipMemcpyToSymbol(HIP_SYMBOL(cudaSimulationParameters), &*_newSimulationParameters, sizeof(SimulationParameters), 0, hipMemcpyHostToDevice));
        _newSimulationParameters.reset();

        if (_cudaSimulationData) {
            _simulationKernels->prepareForSimulationParametersChanges(_settings, getSimulationDataIntern());
        }
    }
}

SimulationData _SimulationCudaFacade::getSimulationDataIntern() const
{
    std::lock_guard lock(_mutexForSimulationData);
    return *_cudaSimulationData;
}
