import argparse  # 导入 argparse 模块，用于解析命令行参数
import datetime  # 导入 datetime 模块，用于处理日期和时间
import os  # 导入 os 模块，用于与操作系统交互
from typing import Any, List  # 从 typing 模块导入 Any 和 List 类型

import pandas as pd  # 导入 pandas 模块，并命名为 pd，用于数据处理
import ray  # 导入 ray 模块，用于并行计算
import torch  # 导入 torch 模块，用于深度学习
from sarathi.config import ParallelConfig  # 从 sarathi.config 模块导入 ParallelConfig 类
from sarathi.model_executor.attention import AttentionBackend  # 从 sarathi.model_executor.attention 模块导入 AttentionBackend 枚举
from tqdm import tqdm  # 从 tqdm 模块导入 tqdm，用于显示进度条

from vidur.profiling.attention.attention_input import AttentionInput  # 从 vidur.profiling.attention.attention_input 模块导入 AttentionInput 类
from vidur.profiling.attention.attention_wrapper import AttentionWrapper  # 从 vidur.profiling.attention.attention_wrapper 模块导入 AttentionWrapper 类
from vidur.profiling.common.model_config import ModelConfig  # 从 vidur.profiling.common.model_config 模块导入 ModelConfig 类
from vidur.profiling.utils import get_attention_input_combinations, get_max_num_blocks  # 从 vidur.profiling.utils 模块导入两个函数

def parse_args():
    parser = argparse.ArgumentParser(description="Attention Profiling")  # 创建一个 ArgumentParser 对象，描述为“Attention Profiling”
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",  # 添加一个布尔参数 --disable_ray，用于禁用 Ray
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",  # 添加一个整型参数 --num_gpus，默认值为 8，表示用于分析的 GPU 数量
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",  # 添加一个字符串参数 --output_dir，默认值为 "profiling_outputs"，表示结果输出目录
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "microsoft/phi-2",
            "internlm/internlm-20b",
            "Qwen/Qwen-72B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "meta-llama/Llama-2-70b-hf",
        ],
        help="Models to profile",  # 添加一个字符串列表参数 --models，默认包含多个模型名称，用于指定要分析的模型
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",  # 添加一个整型列表参数 --num_tensor_parallel_workers，默认值为 [1, 2, 4, 8]，表示要分析的张量并行工作者数量
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum context length model can serve",  # 添加一个整型参数 --max_model_len，默认值为 4096，表示模型可以服务的最大上下文长度
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=4096,
        help="Maximum context length of input",  # 添加一个整型参数 --max_seq_len，默认值为 4096，表示输入的最大上下文长度
    )
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=1,
        help="Maximum decode batch size",  # 添加一个整型参数 --min_batch_size，默认值为 1，表示最小批处理大小
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=128,
        help="Maximum decode batch size",  # 添加一个整型参数 --max_batch_size，默认值为 128，表示最大批处理大小
    )
    parser.add_argument(
        "--profile_only_decode",
        action="store_true",
        help="Only profile the decode",  # 添加一个布尔参数 --profile_only_decode，只分析解码部分
    )
    parser.add_argument(
        "--profile_only_prefill",
        action="store_true",
        help="Only profile the prefill",  # 添加一个布尔参数 --profile_only_prefill，只分析预填充部分
    )
    parser.add_argument(
        "--attention_backend",
        default=AttentionBackend.FLASHINFER,
        choices=[e.value for e in AttentionBackend],
        help="The attention backend to profile (default: %(default)s)",  # 添加一个参数 --attention_backend，默认值为 FLASHINFER，可选值来自 AttentionBackend 枚举
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="Block size for paged attention",  # 添加一个整型参数 --block_size，默认值为 16，表示分页注意力的块大小
    )
    args = parser.parse_args()  # 解析命令行参数

    args.output_dir = f"{args.output_dir}/attention/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"  # 更新输出目录，添加时间戳
    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录，如果已存在则忽略

    return args  # 返回解析后的参数

def profile_model(
    args: argparse.Namespace,
    model: str,
    num_tensor_parallel_workers: int,
    input_combinations: List[AttentionInput],
    max_num_blocks: int,
    dtype: torch.dtype,
    pbar: Any,
):
    model_config = ModelConfig.from_model_name(model)  # 根据模型名称创建模型配置
    parallel_config = ParallelConfig(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )  # 创建并行配置，设置张量并行大小和流水线并行大小

    promises = []  # 初始化 promises 列表，用于存储异步任务
    all_results = []  # 初始化 all_results 列表，用于存储所有结果

    model_wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        AttentionWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})  # 创建一个 Ray 远程 actor，用于封装 AttentionWrapper，并设置环境变量

    model_wrappers = [
        model_wrapper_actor.remote(
            model_config,
            parallel_config,
            max_num_blocks,
            args.max_model_len,
            args.block_size,
            args.attention_backend,
            dtype,
        )
        for _ in range(args.num_gpus)
    ]  # 根据 GPU 数量创建多个 model_wrapper 实例

    for attention_input in input_combinations:  # 遍历所有注意力输入组合
        worker_id = len(promises)  # 获取当前 worker 的 ID
        promise = model_wrappers[worker_id].profile.remote(attention_input)  # 异步调用 profile 方法
        promises.append(promise)  # 将 promise 添加到 promises 列表中

        if len(promises) >= args.num_gpus:  # 如果 promises 的长度达到 GPU 数量
            results = ray.get(promises)  # 获取所有 promises 的结果
            all_results.extend(results)  # 将结果添加到 all_results 列表中
            promises = []  # 清空 promises 列表

        pbar.update(1)  # 更新进度条

    results = ray.get(promises)  # 获取剩余 promises 的结果
    all_results.extend(results)  # 将剩余结果添加到 all_results 列表中

    # 过滤掉所有 None 结果
    all_results = list(filter(None, all_results))  # 过滤掉 None 值

    df = pd.DataFrame(all_results)  # 将所有结果转换为 DataFrame
    # time_stats 列是一个字典，因此需要递归展开为多列，并添加前缀
    df = (
        pd.json_normalize(df["time_stats"])  # 展开 time_stats 列中的字典
        .add_prefix("time_stats.")  # 为展开的列添加前缀
        .join(df.drop(columns=["time_stats"]))  # 将展开的列与原 DataFrame 其他列合并
    )
    return df  # 返回处理后的 DataFrame

def main():
    args = parse_args()  # 解析命令行参数

    # 如果没有禁用 Ray，则初始化 Ray
    if not args.disable_ray:
        ray.init(num_gpus=args.num_gpus,_temp_dir="/mnt/fth/software5/ray_tmp_fth/tmp")  # 初始化 Ray，指定 GPU 数量和临时目录

    dtype = torch.float16  # 设置数据类型为 float16
    input_combinations = get_attention_input_combinations(
        args.max_seq_len,
        args.min_batch_size,
        args.max_batch_size,
        args.profile_only_prefill,
        args.profile_only_decode,
    )  # 获取所有注意力输入组合

    total_combos = {}  # 初始化 total_combos 字典，用于存储每个模型和并行工作者数量的输入组合
    max_num_blocks_dict = {}  # 初始化 max_num_blocks_dict 字典，用于存储每个模型和并行工作者数量的最大块数
    for model in args.models:  # 遍历所有模型
        model_config = ModelConfig.from_model_name(model)  # 获取模型配置
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:  # 遍历所有并行工作者数量
            max_num_blocks = get_max_num_blocks(
                model_config,
                ParallelConfig(
                    tensor_parallel_size=num_tensor_parallel_workers,
                    pipeline_parallel_size=1,
                ),
                args.block_size,
                dtype,
            )  # 计算最大块数
            max_num_blocks_dict[(model, num_tensor_parallel_workers)] = max_num_blocks  # 存储最大块数
            total_combos[(model, num_tensor_parallel_workers)] = list(
                filter(
                    lambda input_combination: input_combination.is_under_memory_limit(
                        max_num_blocks * args.block_size
                    ),
                    input_combinations,
                )
            )  # 过滤输入组合，确保在内存限制下

    pbar = tqdm(total=sum(len(v) for v in total_combos.values()))  # 创建一个进度条，总数为所有组合的总和

    for model in args.models:  # 遍历所有模型
        result_df = pd.DataFrame()  # 初始化结果 DataFrame
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:  # 遍历所有并行工作者数量
            result_df = pd.concat(
                [
                    result_df,
                    profile_model(
                        args,
                        model,
                        num_tensor_parallel_workers,
                        total_combos[(model, num_tensor_parallel_workers)],
                        max_num_blocks_dict[(model, num_tensor_parallel_workers)],
                        dtype,
                        pbar,
                    ),
                ]
            )  # 分析模型并合并结果
        # 模型名称可能包含 '/', 因此创建相应的目录
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)  # 创建模型对应的输出目录
        result_df.to_csv(f"{args.output_dir}/{model}/attention.csv", index=False)  # 将结果保存为 CSV 文件

if __name__ == "__main__":
    main()  # 执行主函数
