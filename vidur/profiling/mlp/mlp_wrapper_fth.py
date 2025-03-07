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

WARMUP_STEPS = 2  # 定义预热步数，用于在正式性能分析前进行模型预热
ACTIVE_STEPS = 20  # 定义正式性能分析的步数


class MlpWrapper:
    def __init__(
        self,
        model_config: ModelConfig,  # 模型配置对象
        num_tensor_parallel_workers: int,  # 张量并行工作线程数
        profile_method: str,  # 性能分析方法（字符串形式）
        rank: int,  # 当前线程或进程的rank值
        output_dir: str,  # 输出目录路径
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

    @torch.inference_mode()  # 使用推理模式上下文管理器，禁用梯度计算以提高性能
    def profile(self, num_tokens: int):  # 定义性能分析方法，接收token数量作为参数
        vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers  # 计算每个张量并行工作线程的词汇表范围
        input_ids = torch.randint(  # 生成随机输入token ID
            low=0,  # 最小值为0
            high=vocab_range,  # 最大值为词汇表范围
            size=(num_tokens,),  # 生成的token数量
            device="cuda",  # 在CUDA设备上生成
            dtype=torch.long,  # 数据类型为long
        )
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)  # 生成位置编码

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:  # 如果使用RECORD_FUNCTION方法
            # 预运行模型一次，确保捕获的图不包含初始基准测试（如Triton自动调优）的内核启动
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()  # 确保所有CUDA操作完成

            self.timer_stats_store.clear_stats()  # 清除时间统计信息

            record_function_tracer = RecordFunctionTracer(self.output_dir)  # 初始化记录函数跟踪器

            with record_function_tracer:  # 使用记录函数跟踪器上下文管理器
                self.model(
                    input_ids,
                    positions,
                )  # 再次运行模型以捕获函数调用跟踪

            time_stats = record_function_tracer.get_operation_time_stats()  # 获取操作时间统计信息
        else:  # 如果使用其他性能分析方法
            for _ in range(WARMUP_STEPS):  # 进行预热步数的模型运行
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()  # 确保所有CUDA操作完成

            self.timer_stats_store.clear_stats()  # 清除时间统计信息

            for _ in range(ACTIVE_STEPS):  # 进行正式性能分析步数的模型运行
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()  # 确保所有CUDA操作完成

            time_stats = self.timer_stats_store.get_stats()  # 获取时间统计信息

        stats = {  # 构造性能分析结果字典
            "time_stats": time_stats,  # 时间统计信息
            "n_head": self.model_config.num_q_heads,  # 查询头数量
            "n_kv_head": self.model_config.num_kv_heads,  # 键值头数量
            "n_embd": self.model_config.embedding_dim,  # 嵌入维度
            "n_expanded_embd": self.model_config.mlp_hidden_dim,  # MLP隐藏层维度
            "vocab_size": self.model_config.vocab_size,  # 词汇表大小
            "use_gated_mlp": self.model_config.use_gated_mlp,  # 是否使用门控MLP
            "num_tokens": num_tokens,  # token数量
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,  # 张量并行工作线程数
        }
        self.timer_stats_store.clear_stats()  # 再次清除时间统计信息

        return stats  # 返回性能分析结果
