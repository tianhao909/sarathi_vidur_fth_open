import os  # 导入os模块，用于处理文件和目录操作

import sarathi.metrics.cuda_timer  # 导入sarathi库中的cuda_timer模块，用于CUDA时间测量
import torch  # 导入PyTorch库，用于深度学习模型的构建和操作

from vidur.profiling.common.cuda_timer import CudaTimer  # 从vidur库中导入CudaTimer类，用于CUDA时间测量

# 将sarathi库中的CudaTimer类替换为vidur库中的CudaTimer实现（猴子补丁）
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.weight_utils import initialize_dummy_weights  # 导入initialize_dummy_weights函数，用于初始化模型权重

from vidur.profiling.common.model_config import ModelConfig  # 导入ModelConfig类，用于定义模型配置
from vidur.profiling.common.timer_stats_store import TimerStatsStore  # 导入TimerStatsStore类，用于存储时间统计信息
from vidur.profiling.mlp.mlp_impl import GPTModel  # 导入GPTModel类，表示多层感知机模型的实现
from vidur.profiling.utils import ProfileMethod  # 导入ProfileMethod枚举类，用于指定性能分析方法
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer  # 导入RecordFunctionTracer类，用于记录函数调用跟踪


from math import ceil  # 导入ceil函数，用于向上取整
from typing import List  # 导入List类型，用于类型注解

import numpy as np  # 导入numpy库，用于数值计算
# import sarathi.metrics.cuda_timer  # 导入sarathi库中的cuda_timer模块，用于CUDA时间测量
# import torch  # 导入PyTorch库，用于深度学习计算

# from vidur.profiling.common.cuda_timer import CudaTimer  # 导入自定义的CudaTimer类，用于CUDA时间测量

# 将sarathi库中的CudaTimer类替换为自定义的CudaTimer类，实现猴子补丁
# sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.config import ParallelConfig  # 导入ParallelConfig类，用于并行配置
from sarathi.model_executor.attention import (  # 导入注意力相关的类和函数
    AttentionBackend,  # 注意力后端枚举类
    get_attention_wrapper,  # 获取注意力包装器的函数
    set_attention_backend,  # 设置注意力后端的函数
)

from vidur.profiling.attention.attention_input import AttentionInput  # 导入AttentionInput类，用于注意力输入
from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy  # 导入SequenceMetadataProxy类，用于序列元数据代理
# from vidur.profiling.common.model_config import ModelConfig  # 导入ModelConfig类，用于模型配置
# from vidur.profiling.common.timer_stats_store import TimerStatsStore  # 导入TimerStatsStore类，用于存储时间统计信息


WARMUP_STEPS = 2  # 定义预热步数，用于在正式性能分析前进行模型预热
ACTIVE_STEPS = 20  # 定义正式性能分析的步数


        # model_wrapper_actor.remote(
        #     model_config,
        #     num_tensor_parallel_workers,
        #     args.profile_method,
        #     rank,
        #     args.output_dir,
        #     parallel_config,
        #     max_num_blocks,
        #     args.max_model_len,
        #     args.block_size,
        #     args.attention_backend,
        #     dtype,
        # )

# class MlpWrapper:
class SarathiWrapper:
    # def __init__(
    #     self,
    #     model_config: ModelConfig,  # 模型配置对象
    #     num_tensor_parallel_workers: int,  # 张量并行工作线程数
    #     profile_method: str,  # 性能分析方法（字符串形式）
    #     rank: int,  # 当前线程或进程的rank值
    #     output_dir: str,  # 输出目录路径
    # ):
    def __init__(
        self,
        model_config: ModelConfig,  # 模型配置对象
        num_tensor_parallel_workers: int,  # 张量并行工作线程数
        profile_method: str,  # 性能分析方法（字符串形式）
        rank: int,  # 当前线程或进程的rank值
        output_dir: str,  # 输出目录路径
        parallel_config: ParallelConfig,  # 并行配置对象
        max_num_blocks: int,  # 最大块数量
        max_model_len: int,  # 最大模型长度
        block_size: int,  # 块大小
        attention_backend: AttentionBackend,  # 注意力后端
        dtype: torch.dtype,  # 数据类型
    ):
        super().__init__()  # 调用父类构造函数

        # 初始化时间统计存储对象，传入性能分析方法
        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.model_config = model_config  # 保存模型配置
        self.num_tensor_parallel_workers = num_tensor_parallel_workers  # 保存张量并行工作线程数
        self.profile_method = profile_method  # 保存性能分析方法
        self.rank = rank  # 保存当前rank值
        self.output_dir = output_dir  # 保存输出目录路径
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)  # 创建性能分析跟踪文件的目录

        # 初始化GPT模型，传入模型配置、张量并行工作线程数以及步数
        self.model = GPTModel(
            model_config,
            num_tensor_parallel_workers,
            (
                ACTIVE_STEPS  # 如果使用RECORD_FUNCTION方法，则步数为ACTIVE_STEPS，否则为1
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                else 1
            ),
        )
        initialize_dummy_weights(self.model)  # 初始化模型的虚拟权重
        self.model = self.model.to(dtype=torch.float16).cuda().eval()  # 将模型转换为float16精度，并移动到CUDA设备上，设置为评估模式

        ##### fth att
        # self.time_stats_store = TimerStatsStore(profile_method="kineto")  # 初始化时间统计存储对象

        self._model_config = model_config  # 存储模型配置
        self._parallel_config = parallel_config  # 存储并行配置
        self._dtype = dtype  # 存储数据类型
        self._device = torch.device("cuda")  # 设置设备为CUDA

        self._max_model_len = max_model_len  # 存储最大模型长度
        self._n_worker_q_heads = self._model_config.get_num_q_heads(  # 获取每个worker的查询头数量
            self._parallel_config
        )
        self._n_worker_kv_heads = self._model_config.get_num_kv_heads(  # 获取每个worker的键值头数量
            self._parallel_config
        )
        self._head_dim = self._model_config.get_head_size()  # 获取每个头的维度

        self._block_size = block_size  # 存储块大小

        self._attention_backend = attention_backend  # 存储注意力后端
        set_attention_backend(attention_backend)  # 设置注意力后端
        get_attention_wrapper().init(  # 初始化注意力包装器
            self._model_config,
            self._parallel_config,
            self._block_size,
            self._device,
        )
        self._max_blocks_per_sequence = ceil(max_model_len / self._block_size)  # 计算每个序列的最大块数
        # 创建并复用大的KV张量
        self.max_num_blocks = max_num_blocks  # 存储最大块数量
        self.kv_cache = get_attention_wrapper().get_cache_block(  # 获取缓存块
            self.max_num_blocks, dtype=self._dtype, device=self._device
        )
        


    # mlp profile
    @torch.inference_mode()  # 使用推理模式上下文管理器，禁用梯度计算以提高性能
    # def profile(self, num_tokens: int):  # 定义性能分析方法，接收token数量作为参数
    # fth mlp+att
    def profile(
        self, 
        num_tokens: int, 
        attention_input: AttentionInput,  # 注意力输入对象
    ):  # 定义性能分析方法，接收token数量作为参数
        
        assert attention_input.is_valid(self._max_model_len)  # fth att 确保输入有效

        # mlp === 共享部分：生成输入数据 ===
        vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers  # mlp 计算每个张量并行工作线程的词汇表范围
        input_ids = torch.randint(  # mlp生成随机输入token ID
            low=0,  # 最小值为0
            high=vocab_range,  # 最大值为词汇表范围
            size=(num_tokens,),  # 生成的token数量
            device="cuda",  # 在CUDA设备上生成
            dtype=torch.long,  # 数据类型为long
        )
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)  # mlp 生成位置编码

        # 获取 Attention 的输入张量
        seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(  # 获取输入张量
            attention_input,
        )

        # === 预热阶段 ===
        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:  # mlp 如果使用RECORD_FUNCTION方法
            # mlp 预运行模型一次，确保捕获的图不包含初始基准测试（如Triton自动调优）的内核启动
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()  # mlp确保所有CUDA操作完成

            self.timer_stats_store.clear_stats()  # mlp清除时间统计信息

            record_function_tracer = RecordFunctionTracer(self.output_dir)  # mlp初始化记录函数跟踪器

            with record_function_tracer:  # mlp使用记录函数跟踪器上下文管理器
                self.model(
                    input_ids,
                    positions,
                )  # mlp再次运行模型以捕获函数调用跟踪

            time_stats = record_function_tracer.get_operation_time_stats()  # mlp获取操作时间统计信息
        else:  # mlp如果使用其他性能分析方法
            for _ in range(WARMUP_STEPS):  # mlp进行预热步数的模型运行
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()  # mlp确保所有CUDA操作完成

            self.timer_stats_store.clear_stats()  # mlp清除时间统计信息

            for _ in range(ACTIVE_STEPS):  # mlp进行正式性能分析步数的模型运行
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()  # mlp确保所有CUDA操作完成

            time_stats = self.timer_stats_store.get_stats()  # mlp获取时间统计信息

        stats = {  # mlp构造性能分析结果字典
            "time_stats": time_stats,  # mlp时间统计信息
            "n_head": self.model_config.num_q_heads,  # mlp查询头数量
            "n_kv_head": self.model_config.num_kv_heads,  # mlp键值头数量
            "n_embd": self.model_config.embedding_dim,  # mlp嵌入维度
            "n_expanded_embd": self.model_config.mlp_hidden_dim,  # MLP隐藏层维度
            "vocab_size": self.model_config.vocab_size,  # mlp词汇表大小
            "use_gated_mlp": self.model_config.use_gated_mlp,  # 是否使用门控MLP
            "num_tokens": num_tokens,  # token数量
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,  # mlp张量并行工作线程数
        }
        self.timer_stats_store.clear_stats()  # mlp再次清除时间统计信息

        # return stats  # fth mlp注释返回性能分析结果
    
        ### att
        # 批量大小在预填充阶段始终为1，在解码阶段可以不同
        assert attention_input.is_valid(self._max_model_len)  # 确保输入有效

        seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(  # 获取输入张量
            attention_input,
        )
        get_attention_wrapper().begin_forward(seq_metadata_list)  # att开始前向传播

        for _ in range(WARMUP_STEPS):  # att预热步骤
            get_attention_wrapper().forward(query, key, value, kv_cache)  # att前向传播
        torch.cuda.synchronize()  # att同步CUDA设备

        self.time_stats_store.clear_stats()  # att清除时间统计信息

        for _ in range(ACTIVE_STEPS):  # att活跃步骤
            get_attention_wrapper().forward(query, key, value, kv_cache)  # att前向传播
        torch.cuda.synchronize()  # att同步CUDA设备

        get_attention_wrapper().end_forward()  # att结束前向传播

        return {  # 返回性能分析结果
            "time_stats": self.time_stats_store.get_stats(),  # att时间统计信息
            "n_embd": self._model_config.embedding_dim,  # att嵌入维度
            "n_q_head": self._model_config.num_q_heads,  # att查询头数量
            "n_kv_head": self._model_config.num_kv_heads,  # att键值头数量
            "block_size": self._block_size,  # att块大小
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,  # att张量并行worker数量
            "max_model_len": self._max_model_len,  # att最大模型长度
            "batch_size": attention_input.batch_size,  # att批量大小
            "prefill_chunk_size": attention_input.prefill_chunk_size,  # att预填充块大小
            "kv_cache_size": attention_input.kv_cache_size,  # attKV缓存大小
            "is_prefill": attention_input.is_prefill,  # att是否为预填充阶段
            "attention_backend": self._attention_backend,  # att注意力后端
        }


    # # att profile
    # @torch.inference_mode()  # 推理模式装饰器，禁用梯度计算
    # def profile(  # 性能分析方法
    #     self,
    #     attention_input: AttentionInput,  # 注意力输入对象
    # ):
    #     # 批量大小在预填充阶段始终为1，在解码阶段可以不同
    #     assert attention_input.is_valid(self._max_model_len)  # 确保输入有效

    #     seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(  # 获取输入张量
    #         attention_input,
    #     )
    #     get_attention_wrapper().begin_forward(seq_metadata_list)  # 开始前向传播

    #     for _ in range(WARMUP_STEPS):  # 预热步骤
    #         get_attention_wrapper().forward(query, key, value, kv_cache)  # 前向传播
    #     torch.cuda.synchronize()  # 同步CUDA设备

    #     self.time_stats_store.clear_stats()  # 清除时间统计信息

    #     for _ in range(ACTIVE_STEPS):  # 活跃步骤
    #         get_attention_wrapper().forward(query, key, value, kv_cache)  # 前向传播
    #     torch.cuda.synchronize()  # 同步CUDA设备

    #     get_attention_wrapper().end_forward()  # 结束前向传播

    #     return {  # 返回性能分析结果
    #         "time_stats": self.time_stats_store.get_stats(),  # 时间统计信息
    #         "n_embd": self._model_config.embedding_dim,  # 嵌入维度
    #         "n_q_head": self._model_config.num_q_heads,  # 查询头数量
    #         "n_kv_head": self._model_config.num_kv_heads,  # 键值头数量
    #         "block_size": self._block_size,  # 块大小
    #         "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,  # 张量并行worker数量
    #         "max_model_len": self._max_model_len,  # 最大模型长度
    #         "batch_size": attention_input.batch_size,  # 批量大小
    #         "prefill_chunk_size": attention_input.prefill_chunk_size,  # 预填充块大小
    #         "kv_cache_size": attention_input.kv_cache_size,  # KV缓存大小
    #         "is_prefill": attention_input.is_prefill,  # 是否为预填充阶段
    #         "attention_backend": self._attention_backend,  # 注意力后端
    #     }
