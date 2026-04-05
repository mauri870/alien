#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <sstream>

#include "Base/Exceptions.h"
#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"

template< typename T >
void checkAndThrowError(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        hipDeviceReset();
        std::stringstream stream;
        switch (result) {
        case hipError_t::hipErrorInsufficientDriver:
            stream << "Your graphics driver is not compatible with the required HIP version. Please update your AMD graphics driver and restart.";
            break;
        case hipError_t::hipErrorOperatingSystem:
            stream << "An operating system call within the HIP API failed. Please check if your monitor is plugged to the correct graphics card.";
            break;
        case hipError_t::hipErrorNotInitialized:
            stream
                << "HIP could not be initialized. Please check the minimum hardware requirements. If fulfilled please update your AMD graphics driver and "
                   "restart.";
            break;
        case hipError_t::hipErrorOutOfMemory:
            stream << "A HIP error occurred while allocating memory. A possible reason could be that there is not enough memory available.";
            break;
        default: {
            stream << "HIP error.";
        }
            break;
        }
        stream << std::endl << "Location: " << file << ":" << line << " code=" << static_cast<unsigned int>(result) << "(" << hipGetErrorName(result) << ") \"" << func
               << "\"";
        auto text = stream.str();
        log(Priority::Important, text);

        if (hipError_t::hipErrorOutOfMemory == result) {
            throw CudaMemoryAllocationException(text);
        } else {
            throw CudaException(text);
        }
    }
}

#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define CHECK_FOR_CUDA_ERROR(val) \
    checkAndThrowError( (val), #val, __FILENAME__, __LINE__ )

#define ABORT() __builtin_trap();

#define NEAR_ZERO 0.00001f

#define CHECK(condition) \
    if (!(condition)) { \
        printf("Check failed. File: %s, Line: %d\n", __FILE__, __LINE__); \
        ABORT(); \
    }

#define CUDA_THROW_NOT_IMPLEMENTED() \
    printf("Not implemented error. File: %s, Line: %d\n", __FILE__, __LINE__); \
    ABORT();

#define KERNEL_CALL(func, ...) \
    if (GlobalSettings::getInstance().isDebugMode()) { \
        func<<<gpuSettings.numBlocks, 8>>>(__VA_ARGS__); \
        hipDeviceSynchronize(); \
        CHECK_FOR_CUDA_ERROR(hipGetLastError()); \
    } \
    else { \
        func<<<gpuSettings.numBlocks, 8>>>(__VA_ARGS__); \
    }

#define KERNEL_CALL_1_1(func, ...) \
    if (GlobalSettings::getInstance().isDebugMode()) { \
        func<<<1, 1>>>(__VA_ARGS__); \
        hipDeviceSynchronize(); \
        CHECK_FOR_CUDA_ERROR(hipGetLastError()); \
    } else { \
        func<<<1, 1>>>(__VA_ARGS__); \
    }

#define KERNEL_CALL_MOD(func, threadsPerBlock, ...) \
    if (GlobalSettings::getInstance().isDebugMode()) { \
        func<<<gpuSettings.numBlocks, threadsPerBlock>>>(__VA_ARGS__); \
        hipDeviceSynchronize(); \
        CHECK_FOR_CUDA_ERROR(hipGetLastError()); \
    } else { \
        func<<<gpuSettings.numBlocks, threadsPerBlock>>>(__VA_ARGS__); \
    }
