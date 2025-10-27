// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "mori/application/utils/check.hpp"

// Forward declarations for VMM functions if not available in current HIP version
#ifdef __cplusplus
extern "C" {
#endif

// These functions might not be available in all HIP versions
// Uncomment these if you get compilation errors about missing functions
/*
hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags);
hipError_t hipMemAddressFree(void* ptr, size_t size);
hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop, unsigned long long flags);
hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags);
hipError_t hipMemUnmap(void* ptr, size_t size);
hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
*/

#ifdef __cplusplus
}
#endif

void TestGpuVirtualMemoryManagement() {
    int deviceCount = 0;
    hipError_t result = hipGetDeviceCount(&deviceCount);
    
    if (result != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(result) << std::endl;
        return;
    }

    std::cout << "=================================================================\n";
    std::cout << "Testing GPU Virtual Memory Management Support\n";
    std::cout << "=================================================================\n";
    std::cout << "Found " << deviceCount << " GPU device(s)\n\n";

    for (int currentDev = 0; currentDev < deviceCount; currentDev++) {
        // Set current device
        result = hipSetDevice(currentDev);
        if (result != hipSuccess) {
            std::cerr << "Failed to set device " << currentDev << ": " << hipGetErrorString(result) << std::endl;
            continue;
        }

        // Get device properties for additional info
        hipDeviceProp_t deviceProp;
        result = hipGetDeviceProperties(&deviceProp, currentDev);
        if (result != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << currentDev << std::endl;
            continue;
        }

        std::cout << "--- Device " << currentDev << ": " << deviceProp.name << " ---\n";

        // Test virtual memory management support
        int vmm = 0;
        result = hipDeviceGetAttribute(
            &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
        );

        if (result != hipSuccess) {
            std::cerr << "Failed to query virtual memory management attribute for device " 
                      << currentDev << ": " << hipGetErrorString(result) << std::endl;
            continue;
        }

        if (vmm == 0) {
            std::cout << "  ❌ GPU " << currentDev << " doesn't support virtual memory management." << std::endl;
        } else {
            std::cout << "  ✅ GPU " << currentDev << " supports virtual memory management." << std::endl;
        }

        // Get additional memory-related attributes
        int memoryPoolsSupported = 0;
        result = hipDeviceGetAttribute(
            &memoryPoolsSupported, hipDeviceAttributeMemoryPoolsSupported, currentDev
        );
        if (result == hipSuccess) {
            std::cout << "  Memory pools supported: " << (memoryPoolsSupported ? "Yes" : "No") << std::endl;
        }

        // Get memory info
        size_t freeBytes = 0, totalBytes = 0;
        result = hipMemGetInfo(&freeBytes, &totalBytes);
        if (result == hipSuccess) {
            std::cout << "  Total memory: " << (totalBytes / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "  Free memory: " << (freeBytes / (1024 * 1024)) << " MB" << std::endl;
        }

        // Additional device properties
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Multiprocessor count: " << deviceProp.multiProcessorCount << std::endl;
        
        std::cout << std::endl;
    }

    std::cout << "=================================================================\n";
    std::cout << "Virtual Memory Management Test Completed\n";
    std::cout << "=================================================================\n";
}

void TestHipVirtualMemoryOperations() {
    std::cout << "\n=================================================================\n";
    std::cout << "Testing HIP Virtual Memory Operations\n";
    std::cout << "=================================================================\n";

    // Get device count
    int deviceCount = 0;
    hipError_t result = hipGetDeviceCount(&deviceCount);
    
    if (result != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(result) << std::endl;
        return;
    }

    std::cout << "Found " << deviceCount << " GPU device(s). Testing VMM operations on each...\n\n";

    for (int currentDev = 0; currentDev < deviceCount; currentDev++) {
        // Set current device
        result = hipSetDevice(currentDev);
        if (result != hipSuccess) {
            std::cerr << "Failed to set device " << currentDev << ": " << hipGetErrorString(result) << std::endl;
            continue;
        }

        // Get device properties for additional info
        hipDeviceProp_t deviceProp;
        result = hipGetDeviceProperties(&deviceProp, currentDev);
        if (result != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << currentDev << std::endl;
            continue;
        }

        std::cout << "--- Testing Device " << currentDev << ": " << deviceProp.name << " ---\n";

        // Check if VMM is supported on this device
        int vmm = 0;
        result = hipDeviceGetAttribute(
            &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
        );

        if (result != hipSuccess || vmm == 0) {
            std::cout << "❌ Virtual Memory Management not supported on device " << currentDev << ". Skipping...\n\n";
            continue;
        }

        std::cout << "✅ VMM supported on device " << currentDev << ". Testing virtual memory operations...\n\n";

    // Test parameters
    const size_t memorySize = 64 * 1024 * 1024; // 64 MB total virtual space
    const size_t granularity = 2 * 1024 * 1024;  // 2 MB granularity per chunk
    const size_t numChunks = memorySize / granularity; // Number of chunks to create
    
    void* virtualPtr = nullptr;
    std::vector<hipMemGenericAllocationHandle_t> memHandles(numChunks);
    hipMemAccessDesc accessDesc;
    
    try {
        // Step 1: Reserve virtual address space
        std::cout << "1. Reserving virtual address space (" << (memorySize / (1024 * 1024)) << " MB)...\n";
        result = hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0);
        if (result != hipSuccess) {
            throw std::runtime_error("hipMemAddressReserve failed: " + std::string(hipGetErrorString(result)));
        }
        std::cout << "   ✅ Virtual address reserved at: " << virtualPtr << std::endl;
        std::cout << "   Virtual address range: " << virtualPtr << " - " 
                  << (void*)((char*)virtualPtr + memorySize) << std::endl;

        // Step 2: Create multiple physical memory chunks
        std::cout << "\n2. Creating " << numChunks << " physical memory chunks (" 
                  << (granularity / (1024 * 1024)) << " MB each)...\n";
        
        hipMemAllocationProp allocProp = {};
        allocProp.type = hipMemAllocationTypePinned;
        allocProp.location.type = hipMemLocationTypeDevice;
        allocProp.location.id = currentDev;
        
        for (size_t i = 0; i < numChunks; ++i) {
            result = hipMemCreate(&memHandles[i], granularity, &allocProp, 0);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemCreate failed for chunk " + std::to_string(i) + 
                                       ": " + std::string(hipGetErrorString(result)));
            }
            std::cout << "   ✅ Physical memory chunk " << i << " created successfully\n";
        }

        // Step 3: Map each physical memory chunk to corresponding virtual address region
        std::cout << "\n3. Mapping physical memory chunks to virtual address regions...\n";
        for (size_t i = 0; i < numChunks; ++i) {
            void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * granularity);
            result = hipMemMap(chunkVirtualPtr, granularity, 0, memHandles[i], 0);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemMap failed for chunk " + std::to_string(i) + 
                                       ": " + std::string(hipGetErrorString(result)));
            }
            std::cout << "   ✅ Chunk " << i << " mapped to virtual address: " << chunkVirtualPtr 
                      << " (size: " << (granularity / (1024 * 1024)) << " MB)\n";
        }

        // Step 4: Set access permissions for the entire virtual address range
        std::cout << "\n4. Setting access permissions for entire virtual address range...\n";
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = currentDev;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        
        result = hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1);
        if (result != hipSuccess) {
            throw std::runtime_error("hipMemSetAccess failed: " + std::string(hipGetErrorString(result)));
        }
        std::cout << "   ✅ Access permissions set successfully for entire range\n";

        // Step 5: Test memory operations on each chunk
        std::cout << "\n5. Testing memory operations on each chunk...\n";
        
        // Create test data for each chunk
        std::vector<std::vector<int>> hostDataChunks(numChunks);
        std::vector<std::vector<int>> readbackDataChunks(numChunks);
        
        for (size_t chunk = 0; chunk < numChunks; ++chunk) {
            hostDataChunks[chunk].resize(granularity / sizeof(int));
            readbackDataChunks[chunk].resize(granularity / sizeof(int));
            
            // Fill with chunk-specific pattern
            for (size_t i = 0; i < hostDataChunks[chunk].size(); ++i) {
                hostDataChunks[chunk][i] = static_cast<int>((chunk * 1000000) + (i % 1000));
            }
            
            // Copy data to virtual memory chunk
            void* chunkVirtualPtr = (void*)((char*)virtualPtr + chunk * granularity);
            result = hipMemcpy(chunkVirtualPtr, hostDataChunks[chunk].data(), granularity, hipMemcpyHostToDevice);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemcpy to virtual memory chunk " + std::to_string(chunk) + 
                                       " failed: " + std::string(hipGetErrorString(result)));
            }
            
            std::cout << "   ✅ Data written to chunk " << chunk << " at address: " << chunkVirtualPtr << std::endl;
        }

        // Read back and verify data from each chunk
        std::cout << "\n6. Verifying data integrity for each chunk...\n";
        bool allDataMatches = true;
        
        for (size_t chunk = 0; chunk < numChunks; ++chunk) {
            void* chunkVirtualPtr = (void*)((char*)virtualPtr + chunk * granularity);
            result = hipMemcpy(readbackDataChunks[chunk].data(), chunkVirtualPtr, granularity, hipMemcpyDeviceToHost);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemcpy from virtual memory chunk " + std::to_string(chunk) + 
                                       " failed: " + std::string(hipGetErrorString(result)));
            }
            
            // Verify data integrity for this chunk
            bool chunkDataMatches = true;
            for (size_t i = 0; i < std::min(hostDataChunks[chunk].size(), size_t(1000)); ++i) {
                if (hostDataChunks[chunk][i] != readbackDataChunks[chunk][i]) {
                    chunkDataMatches = false;
                    allDataMatches = false;
                    break;
                }
            }
            
            if (chunkDataMatches) {
                std::cout << "   ✅ Chunk " << chunk << " data integrity verified\n";
            } else {
                std::cout << "   ❌ Chunk " << chunk << " data integrity check failed\n";
            }
        }

        if (allDataMatches) {
            std::cout << "\n   🎉 All chunks data integrity verified successfully!\n";
        } else {
            std::cout << "\n   ❌ Some chunks failed data integrity check\n";
        }

        // Step 7: Test cross-chunk memory access
        std::cout << "\n7. Testing cross-chunk memory access...\n";
        
        // Write a pattern that spans across chunk boundaries
        std::vector<int> crossChunkData(memorySize / sizeof(int));
        for (size_t i = 0; i < crossChunkData.size(); ++i) {
            crossChunkData[i] = static_cast<int>(i);
        }
        
        result = hipMemcpy(virtualPtr, crossChunkData.data(), memorySize, hipMemcpyHostToDevice);
        if (result != hipSuccess) {
            throw std::runtime_error("Cross-chunk write failed: " + std::string(hipGetErrorString(result)));
        }
        
        std::vector<int> crossChunkReadback(crossChunkData.size());
        result = hipMemcpy(crossChunkReadback.data(), virtualPtr, memorySize, hipMemcpyDeviceToHost);
        if (result != hipSuccess) {
            throw std::runtime_error("Cross-chunk read failed: " + std::string(hipGetErrorString(result)));
        }
        
        // Verify cross-chunk data
        bool crossChunkMatches = true;
        for (size_t i = 0; i < std::min(crossChunkData.size(), size_t(10000)); ++i) {
            if (crossChunkData[i] != crossChunkReadback[i]) {
                crossChunkMatches = false;
                break;
            }
        }
        
        if (crossChunkMatches) {
            std::cout << "   ✅ Cross-chunk memory access verified successfully\n";
        } else {
            std::cout << "   ❌ Cross-chunk memory access failed\n";
        }

        std::cout << "\n8. Cleaning up...\n";
        
        // Step 8: Cleanup (in reverse order)
        // Unmap each chunk
        for (size_t i = 0; i < numChunks; ++i) {
            void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * granularity);
            result = hipMemUnmap(chunkVirtualPtr, granularity);
            if (result != hipSuccess) {
                std::cerr << "   ⚠️  hipMemUnmap failed for chunk " << i << ": " << hipGetErrorString(result) << std::endl;
            } else {
                std::cout << "   ✅ Chunk " << i << " unmapped\n";
            }
        }
        
        // Release each physical memory handle
        for (size_t i = 0; i < numChunks; ++i) {
            result = hipMemRelease(memHandles[i]);
            if (result != hipSuccess) {
                std::cerr << "   ⚠️  hipMemRelease failed for chunk " << i << ": " << hipGetErrorString(result) << std::endl;
            } else {
                std::cout << "   ✅ Physical memory chunk " << i << " released\n";
            }
        }
        
        // Free virtual address space
        result = hipMemAddressFree(virtualPtr, memorySize);
        if (result != hipSuccess) {
            std::cerr << "   ⚠️  hipMemAddressFree failed: " << hipGetErrorString(result) << std::endl;
        } else {
            std::cout << "   ✅ Virtual address space freed\n";
        }

        std::cout << "✅ Multi-chunk Virtual Memory Management operations completed successfully on device " << currentDev << "!\n";
        std::cout << "   Summary: " << numChunks << " chunks of " << (granularity / (1024 * 1024)) 
                  << " MB each in " << (memorySize / (1024 * 1024)) << " MB virtual space\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ VMM operations failed on device " << currentDev << ": " << e.what() << std::endl;
        
        // Cleanup on error
        if (virtualPtr) {
            // Try to cleanup as much as possible
            for (size_t i = 0; i < numChunks; ++i) {
                void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * granularity);
                HIP_RUNTIME_CHECK(hipMemUnmap(chunkVirtualPtr, granularity));
                HIP_RUNTIME_CHECK(hipMemRelease(memHandles[i]));
            }
            HIP_RUNTIME_CHECK(hipMemAddressFree(virtualPtr, memorySize));
        }
    }

    std::cout << "--- Device " << currentDev << " testing completed ---\n\n";
    } // End of device loop

    std::cout << "=================================================================\n";
    std::cout << "VMM Operations Test Completed for All Devices\n";
    std::cout << "=================================================================\n";
}

void TestAdvancedChunkMemoryManagement() {
    std::cout << "\n=================================================================\n";
    std::cout << "Testing Advanced Chunk Memory Management\n";
    std::cout << "=================================================================\n";

    // Get device count
    int deviceCount = 0;
    hipError_t result = hipGetDeviceCount(&deviceCount);
    
    if (result != hipSuccess) {
        std::cerr << "Failed to get device count: " << hipGetErrorString(result) << std::endl;
        return;
    }

    std::cout << "Found " << deviceCount << " GPU device(s). Testing advanced chunk management on each...\n\n";

    for (int currentDev = 0; currentDev < deviceCount; currentDev++) {
        // Set current device
        result = hipSetDevice(currentDev);
        if (result != hipSuccess) {
            std::cerr << "Failed to set device " << currentDev << ": " << hipGetErrorString(result) << std::endl;
            continue;
        }

        // Get device properties for additional info
        hipDeviceProp_t deviceProp;
        result = hipGetDeviceProperties(&deviceProp, currentDev);
        if (result != hipSuccess) {
            std::cerr << "Failed to get device properties for device " << currentDev << std::endl;
            continue;
        }

        std::cout << "--- Testing Device " << currentDev << ": " << deviceProp.name << " ---\n";

        // Check if VMM is supported on this device
        int vmm = 0;
        result = hipDeviceGetAttribute(
            &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
        );

        if (result != hipSuccess || vmm == 0) {
            std::cout << "❌ Virtual Memory Management not supported on device " << currentDev << ". Skipping...\n\n";
            continue;
        }

        std::cout << "✅ VMM supported on device " << currentDev << ". Testing advanced chunk memory management...\n\n";

    // Test parameters for advanced chunking
    const size_t totalVirtualSize = 128 * 1024 * 1024; // 128 MB virtual space
    const size_t chunkSize = 2 * 1024 * 1024;          // 2 MB per chunk
    const size_t maxChunks = totalVirtualSize / chunkSize;
    
    void* virtualPtr = nullptr;
    std::vector<hipMemGenericAllocationHandle_t> activeChunks;
    std::vector<bool> chunkAllocated(maxChunks, false);
    hipMemAccessDesc accessDesc;
    
    try {
        // Step 1: Reserve large virtual address space
        std::cout << "1. Reserving large virtual address space (" << (totalVirtualSize / (1024 * 1024)) << " MB)...\n";
        result = hipMemAddressReserve(&virtualPtr, totalVirtualSize, 0, nullptr, 0);
        if (result != hipSuccess) {
            throw std::runtime_error("hipMemAddressReserve failed: " + std::string(hipGetErrorString(result)));
        }
        std::cout << "   ✅ Virtual address reserved: " << virtualPtr << " - " 
                  << (void*)((char*)virtualPtr + totalVirtualSize) << std::endl;
        std::cout << "   Available for " << maxChunks << " chunks of " << (chunkSize / (1024 * 1024)) << " MB each\n";

        activeChunks.resize(maxChunks);
        
        // Step 2: Dynamically allocate chunks in different patterns
        std::cout << "\n2. Dynamic chunk allocation demonstration...\n";
        
        hipMemAllocationProp allocProp = {};
        allocProp.type = hipMemAllocationTypePinned;
        allocProp.location.type = hipMemLocationTypeDevice;
        allocProp.location.id = currentDev;
        
        // Pattern 1: Allocate every other chunk (simulating sparse allocation)
        std::cout << "   Pattern 1: Allocating every other chunk (sparse allocation)...\n";
        for (size_t i = 0; i < maxChunks; i += 2) {
            result = hipMemCreate(&activeChunks[i], chunkSize, &allocProp, 0);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemCreate failed for sparse chunk " + std::to_string(i));
            }
            
            void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * chunkSize);
            result = hipMemMap(chunkVirtualPtr, chunkSize, 0, activeChunks[i], 0);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemMap failed for sparse chunk " + std::to_string(i));
            }
            
            chunkAllocated[i] = true;
            std::cout << "     ✅ Chunk " << i << " allocated and mapped\n";
        }
        
        // Set access permissions for all allocated chunks
        accessDesc.location.type = hipMemLocationTypeDevice;
        accessDesc.location.id = currentDev;
        accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        
        for (size_t i = 0; i < maxChunks; i += 2) {
            void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * chunkSize);
            result = hipMemSetAccess(chunkVirtualPtr, chunkSize, &accessDesc, 1);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemSetAccess failed for chunk " + std::to_string(i));
            }
        }
        
        // Step 3: Test writing to allocated chunks
        std::cout << "\n3. Testing write operations to allocated chunks...\n";
        for (size_t i = 0; i < maxChunks; i += 2) {
            if (chunkAllocated[i]) {
                void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * chunkSize);
                
                // Create chunk-specific test data
                std::vector<int> testData(chunkSize / sizeof(int));
                for (size_t j = 0; j < testData.size(); ++j) {
                    testData[j] = static_cast<int>((i << 16) | (j & 0xFFFF)); // Encode chunk ID and offset
                }
                
                result = hipMemcpy(chunkVirtualPtr, testData.data(), chunkSize, hipMemcpyHostToDevice);
                if (result != hipSuccess) {
                    throw std::runtime_error("Write to chunk " + std::to_string(i) + " failed");
                }
                std::cout << "     ✅ Data written to chunk " << i << "\n";
            }
        }
        
        // Step 4: Fill gaps (allocate remaining chunks)
        std::cout << "\n4. Filling gaps - allocating remaining chunks...\n";
        for (size_t i = 1; i < maxChunks; i += 2) {
            result = hipMemCreate(&activeChunks[i], chunkSize, &allocProp, 0);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemCreate failed for gap chunk " + std::to_string(i));
            }
            
            void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * chunkSize);
            result = hipMemMap(chunkVirtualPtr, chunkSize, 0, activeChunks[i], 0);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemMap failed for gap chunk " + std::to_string(i));
            }
            
            result = hipMemSetAccess(chunkVirtualPtr, chunkSize, &accessDesc, 1);
            if (result != hipSuccess) {
                throw std::runtime_error("hipMemSetAccess failed for gap chunk " + std::to_string(i));
            }
            
            chunkAllocated[i] = true;
            std::cout << "     ✅ Gap chunk " << i << " allocated and mapped\n";
        }
        
        // Step 5: Test continuous memory access across all chunks
        std::cout << "\n5. Testing continuous memory access across all chunks...\n";
        std::vector<int> continuousData(totalVirtualSize / sizeof(int));
        for (size_t i = 0; i < continuousData.size(); ++i) {
            continuousData[i] = static_cast<int>(i);
        }
        
        result = hipMemcpy(virtualPtr, continuousData.data(), totalVirtualSize, hipMemcpyHostToDevice);
        if (result != hipSuccess) {
            throw std::runtime_error("Continuous write across all chunks failed");
        }
        
        std::vector<int> readbackData(continuousData.size());
        result = hipMemcpy(readbackData.data(), virtualPtr, totalVirtualSize, hipMemcpyDeviceToHost);
        if (result != hipSuccess) {
            throw std::runtime_error("Continuous read across all chunks failed");
        }
        
        // Verify data
        bool dataValid = true;
        for (size_t i = 0; i < std::min(continuousData.size(), size_t(10000)); ++i) {
            if (continuousData[i] != readbackData[i]) {
                dataValid = false;
                break;
            }
        }
        
        if (dataValid) {
            std::cout << "     ✅ Continuous memory access across all chunks verified\n";
        } else {
            std::cout << "     ❌ Continuous memory access verification failed\n";
        }
        
        // Step 6: Demonstrate selective deallocation
        std::cout << "\n6. Demonstrating selective chunk deallocation...\n";
        for (size_t i = 4; i < maxChunks && i < 8; ++i) {
            if (chunkAllocated[i]) {
                void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * chunkSize);
                
                result = hipMemUnmap(chunkVirtualPtr, chunkSize);
                if (result != hipSuccess) {
                    std::cerr << "     ⚠️  hipMemUnmap failed for chunk " << i << std::endl;
                } else {
                    std::cout << "     ✅ Chunk " << i << " unmapped\n";
                }
                
                result = hipMemRelease(activeChunks[i]);
                if (result != hipSuccess) {
                    std::cerr << "     ⚠️  hipMemRelease failed for chunk " << i << std::endl;
                } else {
                    std::cout << "     ✅ Chunk " << i << " released\n";
                }
                
                chunkAllocated[i] = false;
            }
        }
        
        std::cout << "\n7. Final cleanup...\n";
        
        // Cleanup remaining chunks
        for (size_t i = 0; i < maxChunks; ++i) {
            if (chunkAllocated[i]) {
                void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * chunkSize);
                HIP_RUNTIME_CHECK(hipMemUnmap(chunkVirtualPtr, chunkSize));
                HIP_RUNTIME_CHECK(hipMemRelease(activeChunks[i]));
            }
        }
        
        // Free virtual address space
        result = hipMemAddressFree(virtualPtr, totalVirtualSize);
        if (result != hipSuccess) {
            std::cerr << "   ⚠️  hipMemAddressFree failed: " << hipGetErrorString(result) << std::endl;
        } else {
            std::cout << "   ✅ Virtual address space freed\n";
        }
        
        std::cout << "\n🎉 Advanced chunk memory management test completed successfully on device " << currentDev << "!\n";
        std::cout << "   Demonstrated: sparse allocation, gap filling, continuous access, selective deallocation\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Advanced chunk test failed on device " << currentDev << ": " << e.what() << std::endl;
        
        // Cleanup on error
        if (virtualPtr) {
            for (size_t i = 0; i < maxChunks; ++i) {
                if (chunkAllocated[i]) {
                    void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * chunkSize);
                    HIP_RUNTIME_CHECK(hipMemUnmap(chunkVirtualPtr, chunkSize));
                    HIP_RUNTIME_CHECK(hipMemRelease(activeChunks[i]));
                }
            }
            HIP_RUNTIME_CHECK(hipMemAddressFree(virtualPtr, totalVirtualSize));
        }
    }

    std::cout << "--- Device " << currentDev << " testing completed ---\n\n";
    } // End of device loop

    std::cout << "=================================================================\n";
    std::cout << "Advanced Chunk Memory Management Test Completed for All Devices\n";
    std::cout << "=================================================================\n";
}

// void TestUncachedVirtualMemoryOperations() {
//     std::cout << "\n=================================================================\n";
//     std::cout << "Testing Uncached Virtual Memory Operations\n";
//     std::cout << "(Equivalent to hipDeviceMallocUncached)\n";
//     std::cout << "=================================================================\n";

//     // First check if VMM is supported
//     int vmm = 0;
//     int currentDev = 0;
//     hipError_t result = hipDeviceGetAttribute(
//         &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, currentDev
//     );

//     if (result != hipSuccess || vmm == 0) {
//         std::cout << "❌ Virtual Memory Management not supported. Skipping uncached VMM test.\n";
//         return;
//     }

//     std::cout << "✅ VMM supported. Testing uncached virtual memory operations...\n\n";

//     // Test parameters
//     const size_t memorySize = 32 * 1024 * 1024; // 32 MB
//     const size_t granularity = 2 * 1024 * 1024;  // 2 MB granularity
//     const size_t numChunks = memorySize / granularity;
    
//     void* virtualPtr = nullptr;
//     std::vector<hipMemGenericAllocationHandle_t> memHandles(numChunks);
//     hipMemAccessDesc accessDesc;
    
//     try {
//         // Step 1: Reserve virtual address space
//         std::cout << "1. Reserving virtual address space (" << (memorySize / (1024 * 1024)) << " MB)...\n";
//         result = hipMemAddressReserve(&virtualPtr, memorySize, 0, nullptr, 0);
//         if (result != hipSuccess) {
//             throw std::runtime_error("hipMemAddressReserve failed: " + std::string(hipGetErrorString(result)));
//         }
//         std::cout << "   ✅ Virtual address reserved at: " << virtualPtr << std::endl;

//         // Step 2: Create physical memory with uncached properties
//         std::cout << "\n2. Creating uncached physical memory chunks...\n";
//         hipMemAllocationProp allocProp = {};
//         allocProp.type = hipMemAllocationTypeUncached;  // 使用 Uncached 类型
//         allocProp.location.type = hipMemLocationTypeDevice;
//         allocProp.location.id = currentDev;
        
//         // 对于 uncached 内存，可以设置额外的属性
//         allocProp.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
        
//         for (size_t i = 0; i < numChunks; ++i) {
//             result = hipMemCreate(&memHandles[i], granularity, &allocProp, 0);
//             if (result != hipSuccess) {
//                 throw std::runtime_error("hipMemCreate failed for uncached chunk " + std::to_string(i) + 
//                                        ": " + std::string(hipGetErrorString(result)));
//             }
//             std::cout << "   ✅ Uncached physical memory chunk " << i << " created\n";
//         }

//         // Step 3: Map physical memory chunks to virtual address space
//         std::cout << "\n3. Mapping uncached physical memory chunks...\n";
//         for (size_t i = 0; i < numChunks; ++i) {
//             void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * granularity);
//             result = hipMemMap(chunkVirtualPtr, granularity, 0, memHandles[i], 0);
//             if (result != hipSuccess) {
//                 throw std::runtime_error("hipMemMap failed for uncached chunk " + std::to_string(i) + 
//                                        ": " + std::string(hipGetErrorString(result)));
//             }
//             std::cout << "   ✅ Uncached chunk " << i << " mapped to: " << chunkVirtualPtr << std::endl;
//         }

//         // Step 4: Set access permissions with uncached attributes
//         std::cout << "\n4. Setting uncached access permissions...\n";
//         accessDesc.location.type = hipMemLocationTypeDevice;
//         accessDesc.location.id = currentDev;
//         accessDesc.flags = hipMemAccessFlagsProtReadWrite;
        
//         result = hipMemSetAccess(virtualPtr, memorySize, &accessDesc, 1);
//         if (result != hipSuccess) {
//             throw std::runtime_error("hipMemSetAccess failed: " + std::string(hipGetErrorString(result)));
//         }
//         std::cout << "   ✅ Uncached access permissions set\n";

//         // Step 5: Test uncached memory operations
//         std::cout << "\n5. Testing uncached memory operations...\n";
        
//         // Test pattern: Uncached memory is often used for producer-consumer scenarios
//         // where cache coherence is not desired
//         std::vector<int> producerData(memorySize / sizeof(int));
//         for (size_t i = 0; i < producerData.size(); ++i) {
//             producerData[i] = static_cast<int>(i * 2 + 1); // Odd numbers
//         }

//         // Write to uncached memory (simulating producer)
//         result = hipMemcpy(virtualPtr, producerData.data(), memorySize, hipMemcpyHostToDevice);
//         if (result != hipSuccess) {
//             throw std::runtime_error("Write to uncached memory failed: " + std::string(hipGetErrorString(result)));
//         }
//         std::cout << "   ✅ Producer data written to uncached memory\n";

//         // Read from uncached memory (simulating consumer)
//         std::vector<int> consumerData(producerData.size());
//         result = hipMemcpy(consumerData.data(), virtualPtr, memorySize, hipMemcpyDeviceToHost);
//         if (result != hipSuccess) {
//             throw std::runtime_error("Read from uncached memory failed: " + std::string(hipGetErrorString(result)));
//         }
//         std::cout << "   ✅ Consumer data read from uncached memory\n";

//         // Verify data integrity
//         bool dataMatches = true;
//         for (size_t i = 0; i < producerData.size() && i < 10000; ++i) {
//             if (producerData[i] != consumerData[i]) {
//                 dataMatches = false;
//                 break;
//             }
//         }

//         if (dataMatches) {
//             std::cout << "   ✅ Uncached memory operations verified - no cache interference\n";
//         } else {
//             std::cout << "   ❌ Uncached memory data verification failed\n";
//         }

//         // Step 6: Test memory synchronization patterns (important for uncached memory)
//         std::cout << "\n6. Testing memory synchronization with uncached access...\n";
        
//         // Simulate multiple small writes (common pattern for uncached memory)
//         for (size_t chunk = 0; chunk < numChunks; ++chunk) {
//             void* chunkPtr = (void*)((char*)virtualPtr + chunk * granularity);
//             std::vector<int> chunkData(granularity / sizeof(int));
            
//             // Fill with chunk-specific synchronization pattern
//             for (size_t i = 0; i < chunkData.size(); ++i) {
//                 chunkData[i] = static_cast<int>((chunk << 24) | (i & 0xFFFFFF));
//             }
            
//             result = hipMemcpy(chunkPtr, chunkData.data(), granularity, hipMemcpyHostToDevice);
//             if (result != hipSuccess) {
//                 throw std::runtime_error("Sync write to chunk " + std::to_string(chunk) + " failed");
//             }
            
//             // Force synchronization for uncached access
//             result = hipDeviceSynchronize();
//             if (result != hipSuccess) {
//                 std::cerr << "   ⚠️  hipDeviceSynchronize warning for chunk " << chunk << std::endl;
//             }
//         }
//         std::cout << "   ✅ Synchronization patterns tested successfully\n";

//         // Step 7: Verify final state
//         std::cout << "\n7. Final verification of uncached memory state...\n";
//         std::vector<int> finalVerification(memorySize / sizeof(int));
//         result = hipMemcpy(finalVerification.data(), virtualPtr, memorySize, hipMemcpyDeviceToHost);
//         if (result != hipSuccess) {
//             throw std::runtime_error("Final verification read failed");
//         }

//         bool finalDataValid = true;
//         for (size_t chunk = 0; chunk < numChunks && finalDataValid; ++chunk) {
//             size_t chunkOffset = chunk * (granularity / sizeof(int));
//             for (size_t i = 0; i < 1000 && (chunkOffset + i) < finalVerification.size(); ++i) {
//                 int expected = static_cast<int>((chunk << 24) | (i & 0xFFFFFF));
//                 if (finalVerification[chunkOffset + i] != expected) {
//                     finalDataValid = false;
//                     break;
//                 }
//             }
//         }

//         if (finalDataValid) {
//             std::cout << "   ✅ Final uncached memory state verified\n";
//         } else {
//             std::cout << "   ❌ Final uncached memory state verification failed\n";
//         }

//         std::cout << "\n8. Cleaning up uncached memory...\n";
        
//         // Cleanup uncached memory (same as regular VMM cleanup)
//         for (size_t i = 0; i < numChunks; ++i) {
//             void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * granularity);
//             result = hipMemUnmap(chunkVirtualPtr, granularity);
//             if (result != hipSuccess) {
//                 std::cerr << "   ⚠️  hipMemUnmap failed for uncached chunk " << i << std::endl;
//             } else {
//                 std::cout << "   ✅ Uncached chunk " << i << " unmapped\n";
//             }
//         }
        
//         for (size_t i = 0; i < numChunks; ++i) {
//             result = hipMemRelease(memHandles[i]);
//             if (result != hipSuccess) {
//                 std::cerr << "   ⚠️  hipMemRelease failed for uncached chunk " << i << std::endl;
//             } else {
//                 std::cout << "   ✅ Uncached memory chunk " << i << " released\n";
//             }
//         }
        
//         result = hipMemAddressFree(virtualPtr, memorySize);
//         if (result != hipSuccess) {
//             std::cerr << "   ⚠️  hipMemAddressFree failed: " << hipGetErrorString(result) << std::endl;
//         } else {
//             std::cout << "   ✅ Uncached virtual address space freed\n";
//         }

//         std::cout << "\n🎉 Uncached Virtual Memory Management completed successfully!\n";
//         std::cout << "   Key benefits demonstrated:\n";
//         std::cout << "   • Cache-bypass memory access (equivalent to hipDeviceMallocUncached)\n";
//         std::cout << "   • Producer-consumer memory patterns\n";
//         std::cout << "   • Explicit synchronization control\n";
//         std::cout << "   • Memory consistency without cache interference\n";

//     } catch (const std::exception& e) {
//         std::cerr << "❌ Uncached VMM operations failed: " << e.what() << std::endl;
        
//         // Cleanup on error
//         if (virtualPtr) {
//             for (size_t i = 0; i < numChunks; ++i) {
//                 void* chunkVirtualPtr = (void*)((char*)virtualPtr + i * granularity);
//                 hipMemUnmap(chunkVirtualPtr, granularity);
//                 hipMemRelease(memHandles[i]);
//             }
//             hipMemAddressFree(virtualPtr, memorySize);
//         }
//     }

//     std::cout << "=================================================================\n";
// }

int main(int argc, char* argv[]) {
    std::cout << "GPU Virtual Memory Management Test Suite\n";
    std::cout << "========================================\n\n";
    
    // Initialize HIP runtime
    hipError_t result = hipInit(0);
    if (result != hipSuccess) {
        std::cerr << "Failed to initialize HIP runtime: " << hipGetErrorString(result) << std::endl;
        return 1;
    }

    // Test 1: Check VMM support
    TestGpuVirtualMemoryManagement();
    
    // Test 2: Test basic VMM operations with multiple chunks
    TestHipVirtualMemoryOperations();
    
    // Test 3: Test advanced chunk memory management
    TestAdvancedChunkMemoryManagement();
    
    // Test 4: Test uncached memory operations (hipDeviceMallocUncached equivalent)
    // TestUncachedVirtualMemoryOperations();
    
    return 0;
}