#ifndef ONESHOT_ALL2ALL_SDMA_CLASS_HPP
#define ONESHOT_ALL2ALL_SDMA_CLASS_HPP

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <memory>

// 包含必要的头文件
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
    application::SymmMemObjPtr flagsObj_;
    std::unique_ptr<uint64_t[], ShmemDeleter> flags_;

    // 禁用拷贝构造函数和赋值运算符
    All2allSdma(const All2allSdma&) = delete;
    All2allSdma& operator=(const All2allSdma&) = delete;

public:
    /**
     * @brief 构造函数，初始化All2allSdma类
     * @param myPe 当前PE的ID
     * @param npes PE总数
     */
    All2allSdma(int myPe, int npes);

    /**
     * @brief 析构函数，清理资源
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
     * @brief 重置标志位（设置为0）
     */
    void resetFlags();
};

} // namespace collective
} // namespace mori

#endif // ONESHOT_ALL2ALL_SDMA_CLASS_HPP
