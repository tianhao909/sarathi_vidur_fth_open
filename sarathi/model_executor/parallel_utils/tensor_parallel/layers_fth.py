# Copyright 2023 The Sarathi team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
from typing import Optional  # 导入Optional类型提示

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数模块
import torch.nn.init as init  # 导入PyTorch的初始化模块
from torch.nn.parameter import Parameter  # 导入Parameter类

from sarathi.logger import init_logger  # 导入日志初始化函数
from sarathi.metrics.cuda_timer import CudaTimer  # 导入CUDA计时器
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)  # 导入获取张量并行rank和world size的函数

from .mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)  # 导入张量并行区域的映射函数
from .utils import VocabUtility, divide  # 导入词汇工具和分割函数

logger = init_logger(__name__)  # 初始化日志记录器


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}  # 定义张量并行属性的默认值


def param_is_not_tensor_parallel_duplicate(param):  # 判断参数是否不是张量并行重复
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (get_tensor_model_parallel_rank() == 0)  # 如果参数具有张量并行属性或rank为0，则返回True


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):  # 设置张量并行属性
    # Make sure the attributes are not set.  # 确保属性未设置
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)  # 确保张量没有这些属性
    # Set the attributes.  # 设置属性
    setattr(tensor, "tensor_model_parallel", is_parallel)  # 设置张量并行标志
    setattr(tensor, "partition_dim", dim)  # 设置分区维度
    setattr(tensor, "partition_stride", stride)  # 设置分区步长


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):  # 如果未设置张量并行属性，则设置默认值
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):  # 如果张量没有该属性
            setattr(tensor, attribute, value)  # 设置默认值

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])  # 遍历并设置默认值


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):  # 复制张量并行属性
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):  # 如果源张量有该属性
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))  # 复制到目标张量

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)  # 遍历并复制属性


class VocabParallelEmbedding(torch.nn.Module):  # 并行词汇嵌入层
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(
        self,
        num_embeddings: int,  # 词汇表大小
        embedding_dim: int,  # 隐藏状态维度
        *,
        init_method=init.xavier_normal_,  # 权重初始化方法
        params_dtype: torch.dtype = None,  # 参数数据类型
        use_cpu_initialization: bool = False,  # 是否使用CPU初始化
        perform_initialization: bool = False,  # 是否执行初始化
        linear_metric_name: Optional[str] = None,  # 线性操作的度量名称
        communication_metric_name: Optional[str] = None,  # 通信操作的度量名称
        reduce_results: Optional[bool] = True,  # 是否对结果进行规约
        world_size: Optional[int] = None,  # 并行世界的大小
        rank: Optional[int] = None,  # 当前进程的rank
    ):
        super(VocabParallelEmbedding, self).__init__()  # 调用父类构造函数
        assert not perform_initialization  # 确保不执行初始化
        assert not use_cpu_initialization  # 确保不使用CPU初始化

        # Keep the input dimensions.  # 保存输入维度
        self.num_embeddings = num_embeddings  # 词汇表大小
        self.embedding_dim = embedding_dim  # 隐藏状态维度
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()  # 如果未指定数据类型，则使用默认类型

        # Set the defaults for compatibility.  # 设置兼容性默认值
        self.padding_idx = None  # 填充索引
        self.max_norm = None  # 最大范数
        self.norm_type = 2.0  # 范数类型
        self.scale_grad_by_freq = False  # 是否按频率缩放梯度
        self.sparse = False  # 是否稀疏
        self._weight = None  # 权重
        self.tensor_model_parallel_size = (
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )  # 获取张量并行的world size
        self.rank = get_tensor_model_parallel_rank() if rank is None else rank  # 获取当前rank
        self.reduce_results = reduce_results  # 是否对结果进行规约
        # Divide the weight matrix along the vocaburaly dimension.  # 按词汇维度划分权重矩阵
        self.vocab_start_index, self.vocab_end_index = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, self.rank, self.tensor_model_parallel_size
            )
        )  # 计算词汇范围
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )  # 每个分区的词汇数量

        self.weight = Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )  # 初始化权重参数

        self._linear_timer = CudaTimer(linear_metric_name)  # 初始化线性操作计时器
        self._communication_timer = CudaTimer(communication_metric_name)  # 初始化通信操作计时器

    def forward(self, input_):  # 前向传播
        if self.tensor_model_parallel_size > 1:  # 如果张量并行大小大于1
            # Build the mask.  # 构建掩码
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )  # 标记超出范围的词汇
            # Mask the input.  # 掩码输入
            masked_input = input_.clone() - self.vocab_start_index  # 调整输入索引
            masked_input[input_mask] = 0  # 将超出范围的索引置为0
        else:
            masked_input = input_  # 如果不并行，直接使用输入
            # Get the embeddings.  # 获取嵌入
        with self._linear_timer:  # 使用线性操作计时器
            output_parallel = F.embedding(
                masked_input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )  # 计算嵌入

        # Mask the output embedding.  # 掩码输出嵌入
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0  # 将超出范围的嵌入置为0
        if self.reduce_results:
            # Reduce across all the model parallel GPUs.  # 在所有模型并行GPU上规约结果
            with self._communication_timer:  # 使用通信操作计时器
                output = reduce_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel  # 如果不规约，直接使用输出
        return output  # 返回结果


class ColumnParallelLinear(torch.nn.Module):  # 列并行线性层
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
    """

    def __init__(
        self,
        input_size,  # 输入维度
        output_size,  # 输出维度
        *,
        bias=True,  # 是否添加偏置
        gather_output=True,  # 是否收集输出
        init_method=init.xavier_normal_,  # 权重初始化方法
        stride=1,  # 步长
        keep_master_weight_for_test=False,  # 是否保留主权重用于测试
        skip_bias_add=False,  # 是否跳过偏置加法
        params_dtype=None,  # 参数数据类型
        use_cpu_initialization=False,  # 是否使用CPU初始化
        perform_initialization=False,  # 是否执行初始化
        linear_metric_name: Optional[str] = None,  # 线性操作的度量名称
        communication_metric_name: Optional[str] = None,  # 通信操作的度量名称
        world_size: Optional[int] = None,  # 并行世界的大小
        layer_id: Optional[int] = None,  # 层ID
    ):
        super(ColumnParallelLinear, self).__init__()  # 调用父类构造函数
        assert not perform_initialization  # 确保不执行初始化
        assert not use_cpu_initialization  # 确保不使用CPU初始化

        # Keep input parameters  # 保存输入参数
        self.input_size = input_size  # 输入维度
        self.output_size = output_size  # 输出维度
        self.gather_output = gather_output  # 是否收集输出
        # Divide the weight matrix along the last dimension.  # 按最后一个维度划分权重矩阵
        self.world_size = (
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )  # 获取张量并行的world size
        self.output_size_per_partition = divide(output_size, self.world_size)  # 每个分区的输出维度
        self.skip_bias_add = skip_bias_add  # 是否跳过偏置加法

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()  # 如果未指定数据类型，则使用默认类型

        # Parameters.  # 参数
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.  # 注意：F.linear计算XA^T+b，因此我们分配转置
        self.create_weights(params_dtype)  # 创建权重

        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype,
                )
            )  # 初始化偏置
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)  # 设置张量并行属性
            # Always initialize bias to zero.  # 始终将偏置初始化为0
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)  # 如果无偏置，注册为None

        self._linear_timer = CudaTimer(linear_metric_name, layer_id=layer_id)  # 初始化线性操作计时器
        self._communication_timer = CudaTimer(
            communication_metric_name, layer_id=layer_id
        )  # 初始化通信操作计时器

    def create_weights(self, dtype: torch.dtype) -> None:  # 创建权重
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
        )  # 初始化权重参数

    def apply_weights(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:  # 应用权重
        with self._linear_timer:  # 使用线性操作计时器
            return F.linear(x, self.weight, bias)  # 计算线性变换

    def forward(self, input_):  # 前向传播
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None  # 如果跳过偏置加法，则偏置为None

        input_parallel = input_  # 输入并行化
        # Matrix multiply.  # 矩阵乘法
        output_parallel = self.apply_weights(input_parallel, bias)  # 应用权重
        if self.gather_output:
            # All-gather across the partitions.  # 在所有分区上收集输出
            # print(f"++++fth Communicating tensor of shape: {output_parallel.shape}")
            with self._communication_timer:  # 使用通信操作计时器
                output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel  # 如果不收集，直接使用输出
        output_bias = self.bias if self.skip_bias_add else None  # 如果跳过偏置加法，则返回偏置
        return output, output_bias  # 返回输出和偏置


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        reduce_results:
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        params_dtype=None,
        use_cpu_initialization=False,
        perform_initialization=False,
        reduce_results=True,
        linear_metric_name: Optional[str] = None,
        communication_metric_name: Optional[str] = None,
        world_size: Optional[int] = None,
        layer_id: Optional[int] = None,
    ):
        super(RowParallelLinear, self).__init__()  # 调用父类构造函数，初始化模块
        assert not perform_initialization  # 确保未启用 perform_initialization 参数
        assert not use_cpu_initialization  # 确保未启用 use_cpu_initialization 参数

        # Keep input parameters  # 保存输入参数
        self.input_size = input_size  # 输入大小（矩阵 A 的第一个维度）
        self.output_size = output_size  # 输出大小（矩阵 A 的第二个维度）
        self.input_is_parallel = input_is_parallel  # 输入是否已经并行化
        self.reduce_results = reduce_results  # 是否对结果进行归约操作
        if params_dtype is None:  # 如果未指定参数数据类型，则使用默认的 torch 数据类型
            params_dtype = torch.get_default_dtype()

        # Divide the weight matrix along the last dimension.  # 按最后一个维度划分权重矩阵
        self.world_size = (  # 获取当前张量模型并行化的世界大小
            get_tensor_model_parallel_world_size() if world_size is None else world_size
        )
        self.input_size_per_partition = divide(input_size, self.world_size)  # 计算每个分区的输入大小
        self.skip_bias_add = skip_bias_add  # 是否跳过偏置加法操作

        self.create_weights(params_dtype)  # 创建权重参数

        if not reduce_results and (bias and not skip_bias_add):  # 如果不归约结果且需要添加偏置，发出警告
            logger.warning(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:  # 如果需要偏置
            self.bias = Parameter(  # 创建偏置参数
                torch.empty(  # 初始化为空张量
                    self.output_size,  # 偏��大小为输出大小
                    device=torch.cuda.current_device(),  # 使用当前 CUDA 设备
                    dtype=params_dtype,  # 使用指定的数据类型
                )
            )

            # Always initialize bias to zero.  # 始终将偏置初始化为零
            with torch.no_grad():  # 在无梯度上下文中操作
                self.bias.zero_()  # 将偏置值设置为零
        else:
            self.register_parameter("bias", None)  # 如果不需要偏置，注册一个空参数

        self._linear_timer = CudaTimer(linear_metric_name, layer_id=layer_id)  # 创建线性操作计时器
        self._communication_timer = CudaTimer(  # 创建通信操作计时器
            communication_metric_name, layer_id=layer_id
        )

    def create_weights(self, dtype: torch.dtype) -> None:  # 创建权重参数的方法
        self.weight = Parameter(  # 创建权重参数
            torch.empty(  # 初始化为空张量
                self.output_size,  # 权重的第一个维度为输出大小
                self.input_size_per_partition,  # 权重的第二个维度为每个分区的输入大小
                device=torch.cuda.current_device(),  # 使用当前 CUDA 设备
                dtype=dtype,  # 使用指定的数据类型
            )
        )

    def apply_weights(self, x: torch.Tensor) -> torch.Tensor:  # 应用权重的方法
        with self._linear_timer:  # 使用线性操作计时器记录时间
            return F.linear(x, self.weight)  # 执行线性变换：Y = XA

    def forward(self, input_):  # 前向传播方法
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.  # 设置反向传播的全归约操作
        if self.input_is_parallel:  # 如果输入已经并行化
            input_parallel = input_  # 直接使用输入
        else:  # 否则将输入分散到张量模型并行区域
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.  # 执行矩阵乘法
        output_parallel = self.apply_weights(input_parallel)  # 应用权重计算输出
        if self.reduce_results and self.world_size > 1:  # 如果需要归约结果且世界大小大于 1
            with self._communication_timer:  # 使用通信操作计时器记录时间
                output_ = reduce_from_tensor_model_parallel_region(output_parallel)  # 对输出进行归约
        else:  # 否则直接使用并行计算的输出
            output_ = output_parallel

        if not self.skip_bias_add:  # 如果不跳过偏置加法
            output = output_ + self.bias if self.bias is not None else output_  # 添加偏置
            output_bias = None  # 偏置输出为空
        else:  # 如果跳过偏置加法
            output = output_  # 输出为计算结果
            output_bias = self.bias  # 偏置输出为偏置参数
        return output, output_bias  # 返回输出和偏置

