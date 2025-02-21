import argparse
import datetime
import itertools
import os
from typing import Any, List

import pandas as pd
import ray
import yaml
from tqdm import tqdm

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile


def parse_args():
    parser = argparse.ArgumentParser(description="MLP Profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
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
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(
        "--profile_method",
        default="record_function",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    args = parser.parse_args()

    args.output_dir = (
        f"{args.output_dir}/mlp/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model(
    args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any
):
    # print(">> fth 调用我啦 /mnt/fth/software5/vidur/vidur/profiling/mlp/main.py")
    model_config = ModelConfig.from_model_name(model)

    promises = []
    all_results = []

    model_wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        MlpWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
        if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
            pbar.update(len(num_tokens_to_profile))
            continue

        model_wrappers = [
            model_wrapper_actor.remote(
                model_config,
                num_tensor_parallel_workers,
                args.profile_method,
                rank,
                args.output_dir,
            )
            for rank in range(args.num_gpus)
        ]
        for num_tokens in num_tokens_to_profile:
            worker_id = len(promises)
            promise = model_wrappers[worker_id].profile.remote(
                num_tokens,
            )
            promises.append(promise)

            if len(promises) >= args.num_gpus:
                results = ray.get(promises)
                all_results.extend(results)
                promises = []

            pbar.update(1)

    results = ray.get(promises)
    all_results.extend(results)

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    return df


def main():
    args = parse_args()
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))
    # print(f'>>fth test arg={args}')
    
    # fth 初始化 Ray
    # if not args.disable_ray:
    #     ray.init(num_gpus=args.num_gpus,_temp_dir="/mnt/fth/software5/ray_tmp_fth/tmp")
    # if not args.disable_ray:
    #     ray.init(_temp_dir="/mnt/fth/software5/ray_tmp_fth/tmp")


    # 确保临时目录存在且有权限
    temp_dir = "/mnt/fth/software5/ray_tmp_fth/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    os.chmod(temp_dir, 0o777)

    # if not args.disable_ray:
    #     try:
    #         # 尝试连接到现有的 Ray 集群
    #         ray.init(_temp_dir=temp_dir)
    #     except ConnectionError:
    #         # 如果没有现有集群，则启动一个新的头节点
    #         print("No existing Ray cluster found. Starting a new head node...")
    #         os.system(f"ray start --head --num-gpus={args.num_gpus} --temp-dir={temp_dir}")
    #         ray.init(_temp_dir=temp_dir)
    # else:
    #     print("Ray is disabled. Running without Ray.")


    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)

    total_combos = itertools.product(
        args.models,
        num_tokens_to_profile,
        args.num_tensor_parallel_workers,
    )


    # # fth 使用 itertools.tee 创建两个独立的迭代器
    # total_combos_copy, total_combos = itertools.tee(total_combos)

    # # fth 遍历复制的迭代器
    # for combo in total_combos_copy:
    #     print(combo)

    # # 现在 total_combos 仍然可以用于其他操作
    # count = 0
    # for combo in total_combos_copy:
    #     print(f">>fth combo{combo}")
    #     count += 1

    # print(f"Total number of combinations: {count}")
    
    # print(f">>fth test Total combos: {len(list(total_combos))} total_combos={total_combos}")
    # total_combos_list = list(total_combos) # fth 转换为列表并打印
    # print(f">>fth test  啥也没有啊  total_combos_list={total_combos_list}")

    # for combo in total_combos_list:
    #     print(f">>fth test this combo={combo}")

    pbar = tqdm(total=len(list(total_combos)))

    for model in args.models:
        result_df = profile_model(
            args,
            model,
            num_tokens_to_profile,
            pbar,
        )
        # model name would contain '/', so create a directory as required
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/mlp.csv", index=False)


if __name__ == "__main__":
    main()
