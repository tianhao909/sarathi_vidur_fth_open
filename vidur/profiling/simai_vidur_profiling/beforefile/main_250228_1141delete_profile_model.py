import argparse  # 导入argparse模块，用于命令行参数解析
import datetime  # 导入datetime模块，用于处理日期和时间
import itertools  # 导入itertools模块，用于迭代操作
import os  # 导入os模块，用于与操作系统交互
from typing import Any, List  # 从typing模块导入Any和List类型提示

import pandas as pd  # 导入pandas库，并简写为pd，用于数据处理
import ray  # 导入ray库，用于并行和分布式处理
import yaml  # 导入yaml库，用于读取和写入YAML文件
from tqdm import tqdm  # 从tqdm库导入tqdm，用于显示进度条

# 导入自定义模块中的类和函数
from vidur.profiling.common.model_config import ModelConfig  # 从common.model_config模块中导入ModelConfig类
# from vidur.profiling.mlp.mlp_wrapper import MlpWrapper  # 从mlp.mlp_wrapper模块中导入MlpWrapper类
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper  # 从mlp.mlp_wrapper模块中导入MlpWrapper类
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile  # 从utils模块中导入ProfileMethod类和get_num_tokens_to_profile函数

import torch  # 导入 torch 模块，用于深度学习
from sarathi.config import ParallelConfig  # 从 sarathi.config 模块导入 ParallelConfig 类
from sarathi.model_executor.attention import AttentionBackend  # 从 sarathi.model_executor.attention 模块导入 AttentionBackend 枚举
from tqdm import tqdm  # 从 tqdm 模块导入 tqdm，用于显示进度条

from vidur.profiling.attention.attention_input import AttentionInput  # 从 vidur.profiling.attention.attention_input 模块导入 AttentionInput 类
from vidur.profiling.attention.attention_wrapper import AttentionWrapper  # 从 vidur.profiling.attention.attention_wrapper 模块导入 AttentionWrapper 类
from vidur.profiling.common.model_config import ModelConfig  # 从 vidur.profiling.common.model_config 模块导入 ModelConfig 类
from vidur.profiling.utils import get_attention_input_combinations, get_max_num_blocks  # 从 vidur.profiling.utils 模块导入两个函数

from vidur.profiling.simai_vidur_profiling.sarathi_wrapper import SarathiWrapper


def parse_args():  # 定义parse_args函数，用于解析命令行参数

    # parser = argparse.ArgumentParser(description="MLP Profiling")  # 创建ArgumentParser对象，并设置描述
    parser = argparse.ArgumentParser(description="Simai_vidur_Profiling")  # 创建ArgumentParser对象，并设置描述

    #####原mlp代码
    parser.add_argument(  # 添加一个命令行参数选项，用于禁用Ray
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置使用的GPU数量
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置输出目录
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置要分析的模型列表
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
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
        ],
        help="Models to profile",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置张量并行工作者的数量
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于设置最大token数量
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(  # 添加一个命令行参数选项，用于选择分析方法
        "--profile_method",
        default="record_function",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )

    #######原atten代码
    # parser.add_argument(
    #     "--disable_ray",
    #     action="store_true",
    #     help="Disable Ray",  # 添加一个布尔参数 --disable_ray，用于禁用 Ray
    # )
    # parser.add_argument(
    #     "--num_gpus",
    #     type=int,
    #     default=8,
    #     help="Number of GPUs to use for profiling",  # 添加一个整型参数 --num_gpus，默认值为 8，表示用于分析的 GPU 数量
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="profiling_outputs",
    #     help="Output directory for profiling results",  # 添加一个字符串参数 --output_dir，默认值为 "profiling_outputs"，表示结果输出目录
    # )
    # parser.add_argument(
    #     "--models",
    #     type=str,
    #     nargs="+",
    #     default=[
    #         "microsoft/phi-2",
    #         "internlm/internlm-20b",
    #         "Qwen/Qwen-72B",
    #         "meta-llama/Llama-2-7b-hf",
    #         "codellama/CodeLlama-34b-Instruct-hf",
    #         "meta-llama/Llama-2-70b-hf",
    #     ],
    #     help="Models to profile",  # 添加一个字符串列表参数 --models，默认包含多个模型名称，用于指定要分析的模型
    # )
    # parser.add_argument(
    #     "--num_tensor_parallel_workers",
    #     type=int,
    #     nargs="+",
    #     default=[1, 2, 4, 8],
    #     help="Number of tensor parallel workers to profile",  # 添加一个整型列表参数 --num_tensor_parallel_workers，默认值为 [1, 2, 4, 8]，表示要分析的张量并行工作者数量
    # )
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


    #######
    args = parser.parse_args()  # 解析命令行参数

    # 根据当前日期和时间更新输出目录，并创建目录
    # args.output_dir = (
    #     f"{args.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # )
    args.output_dir = (
        f"{args.output_dir}/simai_vidur_profiling/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    return args  # 返回解析后的参数


# def profile_model(  # 定义profile_model函数，用于分析给定模型
#     args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any
# ):

def profile_model(
    args: argparse.Namespace,
    model: str,
    num_tensor_parallel_workers: int,
    input_combinations: List[AttentionInput],
    max_num_blocks: int,
    dtype: torch.dtype,
    pbar: Any,
    num_tokens_to_profile: List[int]
):
    

    model_config = ModelConfig.from_model_name(model)  # 根据模型名称创建模型配置对象

    parallel_config = ParallelConfig(
        tensor_parallel_size=num_tensor_parallel_workers,
        pipeline_parallel_size=1,
    )  # fth att 创建并行配置，设置张量并行大小和流水线并行大小

    promises = []  # 创建空列表用于保存异步任务
    all_results = []  # 创建空列表用于保存所有结果


    # model_wrapper_actor = ray.remote(
    #     num_cpus=1,
    #     num_gpus=1,
    # )(
    #     MlpWrapper,
    # ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    # fth混合mlp和atten
    model_wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        SarathiWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    # fth 创建多个模型包裹器实例
    model_wrappers = [
        model_wrapper_actor.remote(
            model_config,
            num_tensor_parallel_workers,
            args.profile_method,
            rank,
            args.output_dir,
            parallel_config,
            max_num_blocks,
            args.max_model_len,
            args.block_size,
            args.attention_backend,
            dtype,
        )
        for rank in range(args.num_gpus)
    ]


    # mlp wrapper
    # model_wrappers = [
    #     model_wrapper_actor.remote(
    #         model_config,
    #         num_tensor_parallel_workers,
    #         args.profile_method,
    #         rank,
    #         args.output_dir,
    #     )
    #     for rank in range(args.num_gpus)
    # ]

    # att wrapper
    # model_wrappers = [
    #     model_wrapper_actor.remote(
    #         model_config,
    #         parallel_config,
    #         max_num_blocks,
    #         args.max_model_len,
    #         args.block_size,
    #         args.attention_backend,
    #         dtype,
    #     )
    #     for _ in range(args.num_gpus)
    # ]  # 根据 GPU 数量创建多个 model_wrapper 实例


    # for num_tokens in num_tokens_to_profile:  # 遍历要分析的token数量
    #     worker_id = len(promises)  # 获取工作者ID
    #     # # fth mlp+att
    #     # def profile(
    #     #     self, 
    #     #     num_tokens: int, 
    #     #     attention_input: AttentionInput,  # 注意力输入对象
    #     # ):  # 定义性能分析方法，接收token数量作为参数

    #     # promise = model_wrappers[worker_id].profile.remote(
    #     #     num_tokens,
    #     # )  # 调用模型的profile方法
    #     promise = model_wrappers[worker_id].profile.remote(
    #         num_tokens,
    #         attention_input,
    #     )  # 调用模型的profile方法
    #     promises.append(promise)  # 添加到异步任务列表

    #     if len(promises) >= args.num_gpus:  # 如果达到GPU限制
    #         results = ray.get(promises)  # 获取异步任务结果
    #         all_results.extend(results)  # 添加到所有结果列表
    #         promises = []  # 清空异步任务列表

    #     pbar.update(1)  # 更新进度条

    for num_tokens in num_tokens_to_profile:  # 遍历要分析的token数量
        for attention_input in input_combinations:  # 遍历所有注意力输入组合

            worker_id = len(promises)  # 获取工作者ID
            # # fth mlp+att
            promise = model_wrappers[worker_id].profile.remote(
                num_tokens,
                attention_input,
            )  # 调用模型的profile方法
            promises.append(promise)  # 添加到异步任务列表

            if len(promises) >= args.num_gpus:  # 如果达到GPU限制
                results = ray.get(promises)  # 获取异步任务结果
                all_results.extend(results)  # 添加到所有结果列表
                promises = []  # 清空异步任务列表

            pbar.update(1)  # 更新进度条


    # for attention_input in input_combinations:  # 遍历所有注意力输入组合
    #     worker_id = len(promises)  # 获取当前 worker 的 ID
    #     promise = model_wrappers[worker_id].profile.remote(attention_input)  # 异步调用 profile 方法
    #     promises.append(promise)  # 将 promise 添加到 promises 列表中

    #     if len(promises) >= args.num_gpus:  # 如果 promises 的长度达到 GPU 数量
    #         results = ray.get(promises)  # 获取所有 promises 的结果
    #         all_results.extend(results)  # 将结果添加到 all_results 列表中
    #         promises = []  # 清空 promises 列表

    #     pbar.update(1)  # 更新进度条

    # for num_tensor_parallel_workers in args.num_tensor_parallel_workers:  # 遍历张量并行工作者数量
    #     if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:  # 判断是否需要张量并行
    #         pbar.update(len(num_tokens_to_profile))  # 更新进度条
    #         continue

    #     # 创建多个模型包裹器实例
    #     model_wrappers = [
    #         model_wrapper_actor.remote(
    #             model_config,
    #             num_tensor_parallel_workers,
    #             args.profile_method,
    #             rank,
    #             args.output_dir,
    #         )
    #         for rank in range(args.num_gpus)
    #     ]
        # for num_tokens in num_tokens_to_profile:  # 遍历要分析的token数量
        #     worker_id = len(promises)  # 获取工作者ID
        #     promise = model_wrappers[worker_id].profile.remote(
        #         num_tokens,
        #     )  # 调用模型的profile方法
        #     promises.append(promise)  # 添加到异步任务列表

        #     if len(promises) >= args.num_gpus:  # 如果达到GPU限制
        #         results = ray.get(promises)  # 获取异步任务结果
        #         all_results.extend(results)  # 添加到所有结果列表
        #         promises = []  # 清空异步任务列表

        #     pbar.update(1)  # 更新进度条

    results = ray.get(promises)  # 获取剩余异步任务结果
    all_results.extend(results)  # 添加到所有结果列表

    df = pd.DataFrame(all_results)  # 将结果转换为DataFrame
    # 将时间统计数据展开为多个列，并添加前缀
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    return df  # 返回结果DataFrame


def main():  # 定义main函数，程序的主入口
    # 作用：调用 parse_args() 函数来解析传递给脚本的命令行参数，并将结果存储在 args 变量中。
    args = parse_args()  # 解析命令行参数 # /disk1/futianhao/software1/vidur/data/model_configs/meta-llama/Llama-2-70b-hf.yml

    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))  # fth mlp 将参数保存到YAML文件

    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)  #  fth mlp 获取分析的token数量

    ### fth att基础 加mlp
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
                        num_tokens_to_profile
                    ),
                ]
            )  # 分析模型并合并结果, fth 增加num_tokens_to_profile

        # 模型名称可能包含 '/', 因此创建相应的目录
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)  # 创建模型对应的输出目录
        result_df.to_csv(f"{args.output_dir}/{model}/attention.csv", index=False)  # 将结果保存为 CSV 文件

    ########### mlp

    # yaml.dump(...)：将字典形式的参数写入到 config.yaml 文件中，使用 YAML 格式
    # yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))  # 将参数保存到YAML文件

    # num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)  # 获取分析的token数量

    # total_combos = itertools.product(  # 生成要分析的所有组合
    #     args.models,
    #     num_tokens_to_profile,
    #     args.num_tensor_parallel_workers,
    # )

    # pbar = tqdm(total=len(list(total_combos)))  # 创建进度条

    # for model in args.models:  # 遍历每个模型
    #     result_df = profile_model(
    #         args,
    #         model,
    #         num_tokens_to_profile,
    #         pbar,
    #     )  # 分析模型并获取结果
    #     # 根据模型名称创建目录
    #     os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
    #     # result_df.to_csv(f"{args.output_dir}/{model}/mlp.csv", index=False)  # 保存分析结果为CSV文件
    #     result_df.to_csv(f"{args.output_dir}/{model}/mlp.csv", index=False)  # 保存分析结果为CSV文件

if __name__ == "__main__":  # 判断是否在主模块中执行
    main()  # 调用main函数