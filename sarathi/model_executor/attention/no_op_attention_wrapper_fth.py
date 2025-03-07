from typing import List, Optional, Tuple  # 从typing模块导入List, Optional, Tuple类型

import torch  # 导入PyTorch库

from sarathi.config import ModelConfig, ParallelConfig  # 从sarathi.config模块导入ModelConfig和ParallelConfig类
from sarathi.core.datatypes.sequence import SequenceMetadata  # 从sarathi.core.datatypes.sequence模块导入SequenceMetadata类
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper  # 从sarathi.model_executor.attention.base_attention_wrapper模块导入BaseAttentionWrapper类


class NoOpAttentionWrapper(BaseAttentionWrapper):  # 定义NoOpAttentionWrapper类，继承自BaseAttentionWrapper
    _inst = None  # 类变量，用于存储单例实例

    def init(  # 定义初始化方法
        self,  # 实例自身
        model_config: ModelConfig,  # 模型配置
        parallel_config: ParallelConfig,  # 并行配置
        block_size: int,  # 块大小
        device: torch.device,  # 设备
    ):
        self.device = device  # 将设备赋值给实例变量

    def get_cache_block(  # 定义获取缓存块的方法
        self,  # 实例自身
        num_blocks: int,  # 块数量
        **kwargs  # 其他关键字参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回类型为两个torch.Tensor的元组
        pass  # 空实现

    def begin_forward(  # 定义开始前向传播的方法
        self,  # 实例自身
        seq_metadata_list: List[SequenceMetadata],  # 序列元数据列表
    ) -> None:  # 无返回值
        pass  # 空实现

    def end_forward(self):  # 定义结束前向传播的方法
        pass  # 空实现

    def forward(  # 定义前向传播的方法
        self,  # 实例自身
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        value: torch.Tensor,  # 值张量
        kv_cache: Tuple[torch.Tensor, torch.Tensor],  # 键值缓存张量的元组
        softmax_scale: float = 1.0,  # softmax缩放因子，默认值为1.0
        layer_id: Optional[int] = None,  # 层ID，可选参数
    ) -> torch.Tensor:  # 返回类型为torch.Tensor
        return torch.empty_like(query, device=self.device)  # 返回与查询张量形状相同的空张量，并指定设备
