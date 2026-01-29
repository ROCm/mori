#ifndef ONESHOT_ALL2ALL_SDMA_CLASS_HPP
#define ONESHOT_ALL2ALL_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <memory>
#include <cstdint>

// Include necessary headers
#include "mori/application/application.hpp"

namespace mori {
namespace collective {

// 在头文件中完整定义ShmemDeleter
struct ShmemDeleter {
    void operator()(void* ptr) const;
};

template <typename T>
class All2allSdma {
private:
    int myPe_;
    int npes_;
    size_t dtype_size_;
    
    // Flag memory
    application::SymmMemObjPtr flagsObj_;
    std::unique_ptr<uint64_t[], ShmemDeleter> flags_;
    
    // Input transit buffer
    void* input_transit_buffer_;
    size_t input_transit_buffer_size_;
    application::SymmMemObjPtr input_transit_buffer_obj_;
    std::unique_ptr<void, ShmemDeleter> input_transit_buffer_ptr_;
    
    // Output transit buffer
    void* output_transit_buffer_;
    size_t output_transit_buffer_size_;
    application::SymmMemObjPtr output_transit_buffer_obj_;
    std::unique_ptr<void, ShmemDeleter> output_transit_buffer_ptr_;
    
    // Disable copy constructor and assignment operator
    All2allSdma(const All2allSdma&) = delete;
    All2allSdma& operator=(const All2allSdma&) = delete;
    
    // Internal methods
    bool ensure_buffer_size(void*& buffer, 
                           std::unique_ptr<void, ShmemDeleter>& buffer_ptr,
                           size_t& current_size,
                           application::SymmMemObjPtr& buffer_obj,
                           size_t required_size,
                           const char* buffer_name);
    
    void copy_input_to_transit(T* input, size_t total_count, hipStream_t stream);
    void copy_output_to_user(T* output, size_t total_count, hipStream_t stream);

public:
    /**
     * @brief Constructor, initializes All2allSdma class
     * @param myPe Current PE ID
     * @param npes Total number of PEs
     * @param transit_buffer_size Transit buffer size in bytes (default 512MB), half for input and half for output
     */
    All2allSdma(int myPe, int npes, size_t transit_buffer_size = 512 * 1024 * 1024);
    
    /**
     * @brief Constructor, specifying input and output transit buffer sizes separately
     * @param myPe Current PE ID
     * @param npes Total number of PEs
     * @param input_buffer_size Input transit buffer size in bytes
     * @param output_buffer_size Output transit buffer size in bytes
     */
    All2allSdma(int myPe, int npes, size_t input_buffer_size, size_t output_buffer_size);

    /**
     * @brief Destructor, cleans up resources
     */
    ~All2allSdma();

    /**
     * @brief 执行All2All SDMA操作
     * @param input 输入数据指针
     * @param output 输出数据指针
     * @param total_count 每个PE的数据元素数量
     * @param stream HIP流
     * @return 执行时间（秒），如果失败返回-1
     */
    double operator()(T* input, T* output, size_t total_count, hipStream_t stream = nullptr);

    /**
     * @brief 获取标志位对称内存对象
     * @return SymmMemObjPtr 标志位内存对象
     */
    application::SymmMemObjPtr getFlagsObj() const { return flagsObj_; }
    
    /**
     * @brief Gets input transit buffer pointer
     */
    void* getInputTransitBuffer() const { return input_transit_buffer_; }
    
    /**
     * @brief Gets input transit buffer size in bytes
     */
    size_t getInputTransitBufferSize() const { return input_transit_buffer_size_; }
    
    /**
     * @brief Gets input transit buffer symmetric memory object
     */
    application::SymmMemObjPtr getInputTransitBufferObj() const { return input_transit_buffer_obj_; }
    
    /**
     * @brief Gets output transit buffer pointer
     */
    void* getOutputTransitBuffer() const { return output_transit_buffer_; }
    
    /**
     * @brief Gets output transit buffer size in bytes
     */
    size_t getOutputTransitBufferSize() const { return output_transit_buffer_size_; }
    
    /**
     * @brief Gets output transit buffer symmetric memory object
     */
    application::SymmMemObjPtr getOutputTransitBufferObj() const { return output_transit_buffer_obj_; }

    /**
     * @brief Resets flags (sets to 0)
     */
    void resetFlags();
};

} // namespace collective
} // namespace mori

#endif // ONESHOT_ALL2ALL_SDMA_CLASS_HPP
