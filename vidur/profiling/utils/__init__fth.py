
import binascii  # 导入 binascii 模块，用于二进制和 ASCII 编码的转换
import enum  # 导入 enum 模块，用于创建枚举类
from itertools import product  # 从 itertools 模块导入 product，用于笛卡尔积
from math import floor  # 从 math 模块导入 floor，用于向下取整
from typing import List  # 从 typing 模块导入 List，用于类型注解

import torch  # 导入 PyTorch 库
from sarathi.config import ParallelConfig  # 从 sarathi.config 模块导入 ParallelConfig

from vidur.profiling.attention.attention_input import AttentionInput  # 从指定模块导入 AttentionInput 类
from vidur.profiling.collectives.collectives_input import CollectivesInput  # 从指定模块导入 CollectivesInput 类
from vidur.profiling.common.model_config import ModelConfig  # 从指定模块导入 ModelConfig 类


class ProfileMethod(enum.Enum):  # 定义一个枚举类 ProfileMethod
    CUDA_EVENT = "cuda_event"  # 定义枚举成员 CUDA_EVENT
    KINETO = "kineto"  # 定义枚举成员 KINETO
    PERF_COUNTER = "perf_counter"  # 定义枚举成员 PERF_COUNTER
    RECORD_FUNCTION = "record_function"  # 定义枚举成员 RECORD_FUNCTION


def get_num_tokens_to_profile(
    max_num_tokens: int,  # 定义函数 get_num_tokens_to_profile，参数为最大令牌数
):
    NUM_TOKENS_SPACE = (  # 定义 NUM_TOKENS_SPACE 为一个包含各种令牌数量的列表
        list([1, 2, 4])  # 添加1, 2, 4
        + list(range(8, 1024, 8))  # 添加8到1024，步长为8
        + list(range(1024, 2 * 1024 + 1, 16))  # 添加1024到2048，步长为16
        + list(range(2 * 1024, 4 * 1024 + 1, 32))  # 添加2048到4096，步长为32
        + list(range(4 * 1024, 8 * 1024 + 1, 64))  # 添加4096到8192，步长为64
        + list(range(8 * 1024, 16 * 1024 + 1, 128))  # 添加8192到16384，步长为128
        + list(range(16 * 1024, 32 * 1024 + 1, 256))  # 添加16384到32768，步长为256
        + list(range(32 * 1024, 64 * 1024 + 1, 512))  # 添加32768到65536，步长为512
        + list(range(64 * 1024, 128 * 1024 + 1, 1024))  # 添加65536到131072，步长为1024
    )
    num_tokens_to_profile = []  # 初始化空列表用于存储要配置的令牌数
    for num_tokens in NUM_TOKENS_SPACE:  # 遍历 NUM_TOKENS_SPACE 中的每个令牌数
        if num_tokens <= max_num_tokens:  # 如果当前令牌数小于等于最大令牌数
            num_tokens_to_profile.append(num_tokens)  # 将其添加到 num_tokens_to_profile 列表中
        else:  # 如果当前令牌数超过最大值
            break  # 终止循环
    num_tokens_to_profile.sort(reverse=True)  # 将列表按降序排序

    return num_tokens_to_profile  # 返回要配置的令牌数列表


def get_attention_batch_sizes_to_profile(min_batch_size: int, max_batch_size: int):  # 定义函数获取要配置的批量大小
    BATCH_SIZE_SPACE = list(range(1, 128 + 1, 1)) + list(range(128, 1024 + 1, 8))  # 定义批量大小空间
    return list(  # 返回经过筛选后的批量大小列表
        filter(
            lambda x: (x >= min_batch_size and x <= max_batch_size), BATCH_SIZE_SPACE  # 过滤出在最小和最大批量大小之间的值
        )
    )


def get_attention_prefill_chunk_sizes_to_profile(max_seq_len: int):  # 定义函数获取要配置的预填充块大小
    # PREFILL_CHUNK_SIZE_SPACE = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3076, 4096, 8192, 16384]
    # PREFILL_CHUNK_SIZE_SPACE = range(128, 128 * 1024, 128)
    PREFILL_CHUNK_SIZE_SPACE = (  # 定义不同范围和步长的预填充块大小
        list(range(64, 128 + 1, 16))  # 添加64到128，步长为16
        + list(range(128, 1024 + 1, 32))  # 添加128到1024，步长为32
        + list(range(1024, 4 * 1024 + 1, 64))  # 添加1024到4096，步长为64
        + list(range(4 * 1024, 16 * 1024 + 1, 128))  # 添加4096到16384，步长为128
        + list(range(16 * 1024, 64 * 1024 + 1, 256))  # 添加16384到65536，步长为256
    )
    prefill_chunk_sizes_to_profile = []  # 初始化空列表用于存储要配置的预填充块大小
    for prefill_chunk_size in PREFILL_CHUNK_SIZE_SPACE:  # 遍历预填充块大小空间
        if prefill_chunk_size <= max_seq_len:  # 如果当前预填充块大小小于等于最大序列长度
            prefill_chunk_sizes_to_profile.append(prefill_chunk_size)  # 添加到列表中
        else:  # 如果超过最大序列长度
            break  # 终止循环
    return prefill_chunk_sizes_to_profile  # 返回要配置的预填充块大小列表


def get_seq_lengths_to_profile(max_seq_len: int):  # 定义函数获取要配置的序列长度
    SEQ_LENGTH_SIZE_SPACE = (  # 定义序列长度空间
        list(range(0, 1024 + 1, 32))  # 添加0到1024，步长为32
        + list(range(1024, 4 * 1024 + 1, 64))  # 添加1024到4096，步长为64
        + list(range(4 * 1024, 64 * 1024 + 1, 256))  # 添加4096到65536，步长为256
    )
    seq_lengths_to_profile = []  # 初始化空列表用于存储要配置的序列长度
    for seq_length in SEQ_LENGTH_SIZE_SPACE:  # 遍历序列长度空间
        if seq_length < max_seq_len:  # 如果当前序列长度小于最大序列长度
            seq_lengths_to_profile.append(seq_length)  # 添加到列表中
        else:  # 如果达到或超过最大序列长度
            break  # 终止循环
    return seq_lengths_to_profile  # 返回要配置的序列长度列表


def get_attention_input_combinations(
    max_seq_len: int,  # 最大序列长度
    min_batch_size: int,  # 最小批量大小
    max_batch_size: int,  # 最大批量大小
    profile_only_prefill: bool,  # 是否仅配置预填充
    profile_only_decode: bool,  # 是否仅配置解码
):
    input_combinations = []  # 初始化输入组合列表
    # Chunked Prefills
    prefill_chunk_sizes_to_profile = get_attention_prefill_chunk_sizes_to_profile(
        max_seq_len  # 获取要配置的预填充块大小
    )
    for prefill_chunk_size in prefill_chunk_sizes_to_profile:  # 遍历预填充块大小
        num_partitions = max_seq_len // prefill_chunk_size  # 计算分区数量
        kv_cache_sizes_to_profile = [  # 生成 KV 缓存大小列表
            partition_index * prefill_chunk_size
            for partition_index in range(num_partitions)
        ]
        input_combinations.extend(  # 添加笛卡尔积组合到输入组合列表
            product([prefill_chunk_size], kv_cache_sizes_to_profile, [1], [True])
        )
    # Full prefills
    prefill_lengths_to_profile = get_seq_lengths_to_profile(max_seq_len)  # 获取要配置的预填充长度
    input_combinations.extend(product(prefill_lengths_to_profile, [0], [1], [True]))  # 添加完整预填充组合
    # Decodes
    kv_cache_sizes_to_profile = get_seq_lengths_to_profile(max_seq_len)  # 获取要配置的 KV 缓存大小
    batch_sizes_to_profile = get_attention_batch_sizes_to_profile(
        min_batch_size, max_batch_size  # 获取要配置的批量大小
    )
    input_combinations.extend(  # 添加解码组合到输入组合列表
        product([0], kv_cache_sizes_to_profile, batch_sizes_to_profile, [False])
    )

    valid_input_combinations = []  # 初始化有效输入组合列表
    for input_combination in input_combinations:  # 遍历所有输入组合
        prefill_chunk_size, kv_cache_size, batch_size, is_prefill = input_combination  # 解包组合

        if is_prefill and profile_only_decode:  # 如果是预填充且仅配置解码
            continue  # 跳过

        if not is_prefill and profile_only_prefill:  # 如果不是预填充且仅配置预填充
            continue  # 跳过

        attention_input = AttentionInput(  # 创建 AttentionInput 实例
            prefill_chunk_size,
            kv_cache_size,
            batch_size,
            is_prefill,
        )

        if attention_input.is_valid(max_seq_len):  # 验证输入是否有效
            valid_input_combinations.append(attention_input)  # 添加到有效输入组合列表
    return valid_input_combinations  # 返回有效输入组合


"""
    For a given model and parallel config, get the maximum number of blocks that can be allocated.
    This doesn't take into account the weights and activations.
"""


def get_max_num_blocks(
    model_config: ModelConfig,  # 模型配置
    parallel_config: ParallelConfig,  # 并行配置
    block_size: int,  # 块大小
    dtype: torch.dtype,  # 数据类型
    gpu_memory_utilization: float = 0.9,  # GPU 内存利用率，默认0.9
    max_pipeline_parallel_size: int = 8,  # 最大管道并行大小，默认8
):
    element_size = torch.randn(1, dtype=dtype).element_size()  # 获取元素大小
    block_memory_size = (  # 计算单个块的内存大小
        2
        * block_size
        * model_config.get_num_kv_heads(parallel_config)  # KV 头数量
        * model_config.get_head_size()  # 每个头的大小
        * element_size  # 元素大小
    )
    assert model_config.num_layers % max_pipeline_parallel_size == 0  # 确保层数能被最大管道并行大小整除
    block_memory_total = block_memory_size * (  # 计算总块内存
        model_config.num_layers // max_pipeline_parallel_size
    )
    return floor(  # 返回可分配的最大块数
        (torch.cuda.mem_get_info()[1] * gpu_memory_utilization) / (block_memory_total)  # 计算可用内存除以每块内存
    )


def get_collectives_sizes_to_profile(max_collective_size: int):  # 定义函数获取要配置的集体操作大小
    COLLECTIVE_SIZE_SPACE = (  # 定义集体操作大小空间
        list(range(1024, 512 * 1024 + 1, 4 * 1024))  # 添加1024到524288，步长为4096
        + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))  # 添加524288到8388608，步长为16384
        + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))  # 添加8388608到67108864，步长为65536
        + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 265 * 1024))  # 添加67108865到536870912，步长为271360
    )
    collectives_size_to_profile = []  # 初始化要配置的集体操作大小列表
    for collectives_size in COLLECTIVE_SIZE_SPACE:  # 遍历集体操作大小空间
        if collectives_size <= max_collective_size:  # 如果当前大小小于等于最大集体大小
            collectives_size_to_profile.append(collectives_size)  # 添加到列表中
        else:  # 如果超过最大大小
            break  # 终止循环
    return collectives_size_to_profile  # 返回要配置的集体操作大小列表


def get_collectives_inputs(
    num_nodes: int,  # 节点数量
    num_workers_per_node_combinations: List[int],  # 每个节点的工作者数量组合
    max_collective_size: int,  # 最大集体操作大小
    collective: str,  # 集体操作类型
    total_gpus_available: int,  # 可用GPU总数
):
    num_workers = []  # 初始化工作者数量列表

    for num_workers_per_node in num_workers_per_node_combinations:  # 遍历每个节点的工作者数量组合
        for _num_nodes in range(1, num_nodes + 1):  # 遍历节点数量
            num_workers.append(num_workers_per_node * _num_nodes)  # 计算总工作者数量并添加到列表

    num_workers = list(set(num_workers))  # 去重工作者数量列表
    collectives_sizes = get_collectives_sizes_to_profile(max_collective_size)  # 获取要配置的集体操作大小

    collectives_inputs = []  # 初始化集体操作输入列表

    for num_workers, num_workers_per_node, collective_size in product(  # 遍历工作者数量、每节点工作者数量和集体操作大小的笛卡尔积
        num_workers, num_workers_per_node_combinations, collectives_sizes
    ):
        collectives_input = CollectivesInput(  # 创建 CollectivesInput 实例
            num_workers, num_workers_per_node, collective_size, collective
        )
        if not collectives_input.is_valid(total_gpus_available, num_nodes):  # 验证输入是否有效
            continue  # 如果无效，跳过

        collectives_inputs.append(collectives_input)  # 添加到集体操作输入列表

    return collectives_inputs  # 返回集体操作输入列表


def get_cpu_overhead_batch_sizes_to_profile(max_batch_size: int):  # 定义函数获取要配置的CPU开销批量大小
    BATCH_SIZE_SPACE = list(range(8, 64 + 1, 8)) + list(range(64, 256 + 1, 16))  # 定义批量大小空间
    batch_size_to_profile = []  # 初始化批量大小列表
    for batch_size in BATCH_SIZE_SPACE:  # 遍历批量大小空间
        if batch_size <= max_batch_size:  # 如果当前批量大小小于等于最大批量大小
            batch_size_to_profile.append(batch_size)  # 添加到列表中
        else:  # 如果超过最大批量大小
            break  # 终止循环
    return batch_size_to_profile  # 返回要配置的批量大小列表


def hex_to_binary(hex_identifier):  # 定义函数将十六进制字符串转换为二进制
    return binascii.unhexlify(hex_identifier)  # 使用 binascii.unhexlify 进行转换
