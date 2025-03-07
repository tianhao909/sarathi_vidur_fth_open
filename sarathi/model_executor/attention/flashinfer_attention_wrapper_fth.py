from typing import List, Optional  # 从typing模块导入List和Optional类型

import torch  # 导入PyTorch库
from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache  # 从flashinfer模块导入BatchPrefillWithPagedKVCacheWrapper类和append_paged_kv_cache函数

from sarathi.config import ModelConfig, ParallelConfig  # 从sarathi.config模块导入ModelConfig和ParallelConfig类
from sarathi.core.datatypes.sequence import SequenceMetadata  # 从sarathi.core.datatypes.sequence模块导入SequenceMetadata类
from sarathi.metrics.constants import OperationMetrics  # 从sarathi.metrics.constants模块导入OperationMetrics常量
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper  # 从sarathi.model_executor.attention.base_attention_wrapper模块导入BaseAttentionWrapper类


class FlashinferAttentionWrapper(BaseAttentionWrapper):  # 定义FlashinferAttentionWrapper类，继承自BaseAttentionWrapper
    _inst = None  # 类变量，用于存储单例实例

    def init(  # 定义初始化方法
        self,
        model_config: ModelConfig,  # 模型配置
        parallel_config: ParallelConfig,  # 并行配置
        block_size: int,  # 块大小
        device: torch.device,  # 设备
    ):
        super().init(model_config, parallel_config, block_size, device)  # 调用父类的初始化方法

        prefill_workspace_buffer = torch.empty(  # 创建预填充工作区缓冲区
            128 * 1024 * 1024, dtype=torch.uint8, device=device  # 大小为128MB，数据类型为uint8，指定设备
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(  # 初始化预填充包装器
            prefill_workspace_buffer, "NHD"  # 传入工作区缓冲区和布局模式"NHD"
        )

        decode_workspace_buffer = torch.empty(  # 创建解码工作区缓冲区
            128 * 1024 * 1024, dtype=torch.uint8, device=device  # 大小为128MB，数据类型为uint8，指定设备
        )
        self.decode_wrapper = BatchPrefillWithPagedKVCacheWrapper(  # 初始化解码包装器
            decode_workspace_buffer, "NHD"  # 传入工作区缓冲区和布局模式"NHD"
        )

        self.is_metadata_initialized = False  # 标记元数据是否已初始化
        self.is_profiling_iteration = False  # 标记是否为性能分析迭代
        self.contains_prefill = False  # 标记是否包含预填充
        self.contains_decode = False  # 标记是否包含解码
        self.num_prefill_tokens = 0  # 预填充的token数量
        self.num_total_tokens = 0  # 总token数量

        self.append_qo_indptr_tensor = None  # 初始化append_qo_indptr_tensor为空
        self.append_kv_page_indices_tensor = None  # 初始化append_kv_page_indices_tensor为空
        self.append_kv_page_indptr_tensor = None  # 初始化append_kv_page_indptr_tensor为空
        self.append_kv_last_page_len_tensor = None  # 初始化append_kv_last_page_len_tensor为空

    def to_int_tensor(self, data: List[int]) -> torch.Tensor:  # 定义将列表转换为整数tensor的方法
        return torch.tensor(data, dtype=torch.int32, device="cuda")  # 创建int32类型的CUDA tensor

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:  # 定义获取缓存块的方法
        return torch.randn(  # 返回随机数生成的tensor
            num_blocks,  # 块数量
            2,  # 第二维大小为2
            self.block_size,  # 第三维大小为块大小
            self.num_kv_heads,  # 第四维大小为KV头的数量
            self.head_dim,  # 第五维大小为头维度
            **kwargs,  # 其他关键字参数
        )

    def begin_forward(  # 定义开始前向传播的方法
        self,
        seq_metadata_list: List[SequenceMetadata],  # 序列元数据列表
    ) -> None:
        # 注释：indptr张量捕获输入张量中查询token的位置。
        # |<---------------------- num_valid_tokens ----------------------------------------------------->|
        # |<--------------- num_prompt_tokens -------------->||<------- num_generation_tokens (M) ------->|
        # |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->||<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
        #
        # 注释：Flashinfer将此布局称为raggedtensor。indptr张量捕获ragged tensor中每个序列的起始位置。
        # indptr张量的长度为序列数量 + 1。我们在一次调用batched prefill kernel时同时执行预填充和解码注意力。
        # prefill_qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        prefill_qo_indptr: List[int] = [0]  # 初始化预填充的qo indptr列表
        decode_qo_indptr: List[int] = [0]  # 初始化解码的qo indptr列表
        # 注释：kv_page_indices张量捕获分配给输入张量中每个token的键值缓存页面。由于每个序列分配的页面数量不固定，因此使用ragged tensor表示。
        prefill_kv_page_indices: List[int] = []  # 初始化预填充的kv页面索引列表
        decode_kv_page_indices: List[int] = []  # 初始化解码的kv页面索引列表
        # 注释：最后一页可能不满，所以需要跟踪最后一页的长度
        prefill_kv_last_page_len: List[int] = []  # 初始化预填充的kv最后页面长度列表
        decode_kv_last_page_len: List[int] = []  # 初始化解码的kv最后页面长度列表
        # 注释：由于prefill_kv_page_indices张量是ragged tensor，我们还需要跟踪prefill_kv_page_indices张量的indptr张量。
        # 该张量捕获ragged tensor中每个序列的起始位置。
        prefill_kv_page_indptr: List[int] = [0]  # 初始化预填充的kv页面indptr列表
        decode_kv_page_indptr: List[int] = [0]  # 初始化解码的kv页面indptr列表

        self.is_profiling_iteration = False  # 重置性能分析迭代标记
        self.is_metadata_initialized = True  # 设置元数据已初始化标记

        self.contains_prefill = False  # 重置是否包含预填充标记
        self.contains_decode = False  # 重置是否包含解码标记

        for seq_metadata in seq_metadata_list:  # 遍历序列元数据列表
            if not seq_metadata.is_prompt:  # 如果序列不是提示
                continue  # 跳过该序列

            # 注释：仅用于性能分析
            if seq_metadata.block_table is None:  # 如果序列的block_table为空
                self.is_profiling_iteration = True  # 设置为性能分析迭代
                # 注释：在内存性能分析期间，块表尚未初始化。
                # 我们将暂时跳过注意力计算。
                return  # 结束方法

            self.contains_prefill = True  # 标记包含预填充

            prompt_chunk_len = seq_metadata.prompt_chunk_len  # 获取提示块的长度
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()  # 获取已处理的提示token数量
            current_total_len = processed_prompt_len + prompt_chunk_len  # 计算当前总长度

            # 注释：qo张量中提示token的indptr
            prefill_qo_indptr.append(prefill_qo_indptr[-1] + prompt_chunk_len)  # 更新预填充qo indptr
            # 注释：计算提示token的kv页面索引
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size  # 计算使用的块数
            prefill_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])  # 添加kv页面索引
            prefill_kv_page_indptr.append(
                prefill_kv_page_indptr[-1] + num_blocks_in_use  # 更新kv页面indptr
            )
            prefill_kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size  # 计算最后一个kv页面的长度
            )

        for seq_metadata in seq_metadata_list:  # 再次遍历序列元数据列表
            if seq_metadata.is_prompt:  # 如果序列是提示
                continue  # 跳过该序列

            if seq_metadata.block_table is None:  # 如果序列的block_table为空
                self.is_profiling_iteration = True  # 设置为性能分析迭代
                return  # 结束方法

            self.contains_decode = True  # 标记包含解码

            context_len = seq_metadata.seq.get_len()  # 获取上下文长度
            # 注释：qo张量中提示token的indptr
            decode_qo_indptr.append(decode_qo_indptr[-1] + 1)  # 更新解码qo indptr
            # 注释：计算提示token的kv页面索引
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size  # 计算使用的块数
            decode_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])  # 添加kv页面索引
            decode_kv_page_indptr.append(decode_kv_page_indptr[-1] + num_blocks_in_use)  # 更新kv页面indptr
            decode_kv_last_page_len.append(
                context_len % self.block_size or self.block_size  # 计算最后一个kv页面的长度
            )

        if self.contains_prefill:  # 如果包含预填充
            self.prefill_wrapper.begin_forward(  # 调用预填充包装器的begin_forward方法
                self.to_int_tensor(prefill_qo_indptr),  # 转换并传递prefill_qo_indptr
                self.to_int_tensor(prefill_kv_page_indptr),  # 转换并传递prefill_kv_page_indptr
                self.to_int_tensor(prefill_kv_page_indices),  # 转换并传递prefill_kv_page_indices
                self.to_int_tensor(prefill_kv_last_page_len),  # 转换并传递prefill_kv_last_page_len
                self.num_q_heads,  # 传递查询头数量
                self.num_kv_heads,  # 传递键值头数量
                self.head_dim,  # 传递头维度
                self.block_size,  # 传递块大小
            )

        if self.contains_decode:  # 如果包含解码
            self.decode_wrapper.begin_forward(  # 调用解码包装器的begin_forward方法
                self.to_int_tensor(decode_qo_indptr),  # 转换并传递decode_qo_indptr
                self.to_int_tensor(decode_kv_page_indptr),  # 转换并传递decode_kv_page_indptr
                self.to_int_tensor(decode_kv_page_indices),  # 转换并传递decode_kv_page_indices
                self.to_int_tensor(decode_kv_last_page_len),  # 转换并传递decode_kv_last_page_len
                self.num_q_heads,  # 传递查询头数量
                self.num_kv_heads,  # 传递键值头数量
                self.head_dim,  # 传递头维度
                self.block_size,  # 传递块大小
            )

        self.num_prefill_tokens = prefill_qo_indptr[-1]  # 设置预填充的token数量
        self.num_total_tokens = self.num_prefill_tokens + len(decode_qo_indptr) - 1  # 计算总token数量

        self.append_qo_indptr_tensor = self.to_int_tensor(  # 创建并设置append_qo_indptr_tensor
            prefill_qo_indptr[:-1]
            + [x + prefill_qo_indptr[-1] for x in decode_qo_indptr]  # 合并预填充和解码的indptr
        )
        self.append_kv_page_indices_tensor = self.to_int_tensor(  # 创建并设置append_kv_page_indices_tensor
            prefill_kv_page_indices + decode_kv_page_indices  # 合并预填充和解码的kv页面索引
        )
        self.append_kv_page_indptr_tensor = self.to_int_tensor(  # 创建并设置append_kv_page_indptr_tensor
            prefill_kv_page_indptr[:-1]
            + [x + prefill_kv_page_indptr[-1] for x in decode_kv_page_indptr]  # 合并预填充和解码的kv页面indptr
        )
        self.append_kv_last_page_len_tensor = self.to_int_tensor(  # 创建并设置append_kv_last_page_len_tensor
            prefill_kv_last_page_len + decode_kv_last_page_len  # 合并预填充和解码的kv最后页面长度
        )

    def end_forward(self):  # 定义结束前向传播的方法
        if self.contains_prefill:  # 如果包含预填充
            self.prefill_wrapper.end_forward()  # 调用预填充包装器的end_forward方法

        if self.contains_decode:  # 如果包含解码
            self.decode_wrapper.end_forward()  # 调用解码包装器的end_forward方法

        self.is_metadata_initialized = False  # 重置元数据已初始化标记

    def forward(  # 定义前向传播的方法
        self,
        query: torch.Tensor,  # 查询张量
        key: torch.Tensor,  # 键张量
        value: torch.Tensor,  # 值张量
        kv_cache: torch.Tensor,  # 键值缓存张量
        softmax_scale: float = 1.0,  # softmax缩放因子，默认值为1.0
        layer_id: Optional[int] = None,  # 层ID，可选参数
    ) -> torch.Tensor:
        assert self.is_metadata_initialized, "Metadata is not initialized."  # 确保元数据已初始化

        if self.is_profiling_iteration:  # 如果是性能分析迭代
            # 注释：在性能分析模式下无需调用注意力
            return torch.zeros_like(query)  # 返回与查询张量相同形状的零张量

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):  # 计时器，用于衡量ATTN_INPUT_RESHAPE操作
            query = query.contiguous().reshape(-1, self.num_q_heads, self.head_dim)  # 重新调整查询张量的形状
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)  # 重新调整键张量的形状
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)  # 重新调整值张量的形状

        output = torch.empty_like(query)  # 创建与查询张量形状相同的空输出张量

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):  # 计时器，用于衡量ATTN_KV_CACHE_SAVE操作
            append_paged_kv_cache(  # 调用append_paged_kv_cache函数
                key,  # 键张量
                value,  # 值张量
                self.append_qo_indptr_tensor,  # qo indptr张量
                kv_cache,  # 键值缓存张量
                self.append_kv_page_indices_tensor,  # kv页面索引张量
                self.append_kv_page_indptr_tensor,  # kv页面indptr张量
                self.append_kv_last_page_len_tensor,  # kv最后页面长度张量
                kv_layout="NHD",  # kv布局模式为"NHD"
            )

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):  # 计时器，用于衡量ATTN_PREFILL操作
            if self.contains_prefill:  # 如果包含预填充
                output[: self.num_prefill_tokens] = self.prefill_wrapper.forward(  # 调用预填充包装器的forward方法，并存储结果到输出张量
                    query[: self.num_prefill_tokens],  # 传入查询张量的预填充部分
                    kv_cache,  # 键值缓存张量
                    pos_encoding_mode="NONE",  # 位置编码模式设置为"NONE"
                    sm_scale=softmax_scale,  # 传入softmax缩放因子
                )

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):  # 计时器，用于衡量ATTN_DECODE操作
            if self.contains_decode:  # 如果包含解码
                output[self.num_prefill_tokens : self.num_total_tokens] = (  # 调用解码包装器的forward方法，并存储结果到输出张量
                    self.decode_wrapper.forward(
                        query[self.num_prefill_tokens : self.num_total_tokens],  # 传入查询张量的解码部分
                        kv_cache,  # 键值缓存张量
                        pos_encoding_mode="NONE",  # 位置编码模式设置为"NONE"
                        sm_scale=softmax_scale,  # 传入softmax缩放因子
                    )
                )

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):  # 计时器，用于衡量ATTN_OUTPUT_RESHAPE操作
            output = output.reshape(-1, self.num_q_heads * self.head_dim)  # 重新调整输出张量的形状

        return output  # 返回输出张量