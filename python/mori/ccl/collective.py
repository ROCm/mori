import numpy as np
from mori import cpp as mori_cpp
from mori import shmem as mori_shme

class All2allSdma:
    """极简的All2All SDMA封装"""

    def __init__(self, my_pe: int, npes: int):
        self.my_pe = my_pe
        self.npes = npes

    def __call__(self,
                 input_data: np.ndarray,
                 output_data: np.ndarray,
                 count: int) -> float:
        """
        执行All2All操作

        Args:
            input_data: 输入数据 (uint32 numpy数组)
            output_data: 输出数据 (uint32 numpy数组)
            count: 每个PE的元素数量

        Returns:
            执行时间(秒)
        """
        # 直接调用C++函数
        return mori_cpp.all2all_sdma(
            self.my_pe,
            self.npes,
            input_data,
            output_data,
            count
        )
