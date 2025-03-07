from abc import ABC, abstractmethod  # 从abc模块导入抽象基类和抽象方法装饰器
from typing import List, Optional, Tuple, Union  # 从typing模块导入类型提示相关类

import torch  # 导入PyTorch库

from sarathi.config import ModelConfig, ParallelConfig  # 从sarathi.config模块导入模型配置和并行配置类
from sarathi.core.datatypes.sequence import SequenceMetadata  # 从sarathi.core.datatypes.sequence模块导入序列元数据类
from sarathi.metrics.constants import OperationMetrics  # 从sarathi.metrics.constants模块导入操作指标常量
from sarathi.metrics.cuda_timer import CudaTimer  # 从sarathi.metrics.cuda_timer模块导入CUDA计时器类


class BaseAttentionWrapper(ABC):  # 定义一个继承自ABC的抽象基类BaseAttentionWrapper
    _inst = None  # 类属性，用于存储单例实例

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
    ):
        self.device = device  # 将设备信息赋值给实例属性device
        self.num_q_heads = model_config.get_num_q_heads(parallel_config)  # 获取查询头的数量并赋值
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)  # 获取键值头的数量并赋值
        self.head_dim = model_config.get_head_size()  # 获取每个头的维度大小并赋值
        self.dtype = model_config.dtype  # 获取数据类型并赋值
        self.block_size = block_size  # 将块大小赋值给实例属性block_size
        self._timers = {}  # 初始化一个空字典用于存储计时器

    """
    对于给定的模型，所有层共享同一个AttentionWrapper实例。
    但是，我们不能为所有层使用一个计时器，因为相同的计时器无法动态开启/关闭。
    因此，我们为每个层单独设置计时器。
    """  # 多行注释，解释计时器的使用策略

    def get_timer(self, operation: OperationMetrics, layer_id: Optional[int] = None):
        if self._timers.get((operation, layer_id)) is None:  # 如果指定操作和层ID的计时器不存在
            self._timers[(operation, layer_id)] = CudaTimer(operation, layer_id)  # 创建并存储一个新的CUDA计时器
        return self._timers.get((operation, layer_id))  # 返回对应的计时器

    @abstractmethod  # 标记以下方法为抽象方法
    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        pass  # 抽象方法占位，无实现

    @classmethod  # 标记以下方法为类方法
    def get_instance(cls):
        if cls._inst is None:  # 如果类属性_inst尚未实例化
            cls._inst = cls()  # 创建类的实例并赋值给_inst
        return cls._inst  # 返回单例实例

    @abstractmethod  # 标记以下方法为抽象方法
    def end_forward(self):
        pass  # 抽象方法占位，无实现

    @abstractmethod  # 标记以下方法为抽象方法
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        pass  # 抽象方法占位，无实现
