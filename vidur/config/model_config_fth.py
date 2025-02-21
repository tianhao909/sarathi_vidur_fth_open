from dataclasses import dataclass, field  # 导入 dataclass 和 field，用于定义数据类
from typing import Any, Dict, Optional     # 导入类型提示相关库

from vidur.config.base_fixed_config import BaseFixedConfig  # 从 vidur 的配置模块导入基础固定配置类
from vidur.logger import init_logger         # 从 vidur 的日志模块导入初始化日志的函数
from vidur.types import ActivationType, NormType  # 从 vidur 的类型模块导入激活函数类型和规范化类型

logger = init_logger(__name__)  # 初始化日志记录器，用于记录日志

@dataclass
class BaseModelConfig(BaseFixedConfig):  # 定义一个基础模型配置类，继承自 BaseFixedConfig
    num_layers: int   # 模型层数
    num_q_heads: int  # 查询头（Q head）的数量
    num_kv_heads: int # 键值头（KV head）的数量
    embedding_dim: int # 输入嵌入层的维度
    mlp_hidden_dim: int # MLP 层的隐藏层维度
    max_position_embeddings: int  # 最大位置嵌入的长度
    use_gated_mlp: bool  # 是否使用门控 MLP（Gated MLP）
    use_bias: bool       # 是否在层中使用偏置（Bias）
    use_qkv_bias: bool   # 是否在 QKV（查询、键、值）层中使用偏置（Bias）
    activation: ActivationType  # 激活函数类型
    norm: NormType      # 规范化层类型（如 LayerNorm、RMSNorm）
    post_attn_norm: bool  # 是否在注意力层之后添加规范化层
    vocab_size: int       # 词汇表大小
    is_neox_style: Optional[bool] = True  # 是否是 Neox 风格的模型架构（可选，默认为 True）
    rope_theta: Optional[float] = None    # RoPE（旋转位置编码）的θ值（可选，默认为 None）
    rope_scaling: Optional[Dict[str, Any]] = None  # RoPE 的扩展参数（可选，默认为 None）
    partial_rotary_factor: float = 1.0     # 部分旋转因子，默认为 1.0
    no_tensor_parallel: bool = False      # 是否禁用张量并行（Tensor Parallelism，可选，默认为 False）

@dataclass
class Llama2ModelConfig(BaseModelConfig):  # 定义 Llama 2 模型的配置类，继承自 BaseModelConfig
    max_position_embeddings: int = 16384  # 最大位置嵌入的长度，默认为 16384
    use_gated_mlp: bool = True            # 是否使用门控 MLP，默认为 True
    use_bias: bool = False                # 是否使用偏置，默认为 False
    use_qkv_bias: bool = False            # 是否在 QKV 层使用偏置，默认为 False
    activation: ActivationType = ActivationType.SILU  # 激活函数类型，默认为 SiLU
    norm: NormType = NormType.RMS_NORM    # 规范化层类型，默认为 RMS 规范化
    post_attn_norm: bool = True           # 是否在注意力层之后添加规范化层，默认为 True
    vocab_size: int = 32768               # 词汇表大小，默认为 32768
    is_neox_style: Optional[bool] = True  # 是否是 Neox 风格，默认为 True
    rope_theta: Optional[float] = 10000   # RoPE 的θ值，默认为 10000
    rope_scaling: Optional[Dict[str, Any]] = None  # RoPE 的扩展参数，默认为 None
    partial_rotary_factor: float = 1.0    # 部分旋转因子，默认为 1.0
    no_tensor_parallel: bool = False      # 是否禁用张量并行，默认为 False

    @staticmethod
    def get_name():  # 静态方法，获取模型名称
        return "meta-llama/Llama-2-Config"  # 返回模型名称的字符串

@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):  # 定义 Codellama 34B 模型的配置类，继承自 Llama2ModelConfig
    num_layers: int = 48  # 模型层数，默认为 48
    num_q_heads: int = 64 # 查询头数量，默认为 64
    num_kv_heads: int = 8 # 键值头数量，默认为 8
    embedding_dim: int = 8192  # 嵌入层维度，默认为 8192
    mlp_hidden_dim: int = 22016 # MLP 层隐藏层维度，默认为 22016
    rope_theta: Optional[float] = 1000000  # RoPE 的θ值，默认为 1000000

    @staticmethod
    def get_name():  # 获取模型名称
        return "codellama/CodeLlama-34b-Instruct-hf"  # 返回模型名称

@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):  # 定义 Llama 2 7B 模型的配置类，继承自 Llama2ModelConfig
    num_layers: int = 32   # 模型层数，默认为 32
    num_q_heads: int = 32  # 查询头数量，默认为 32
    num_kv_heads: int = 32 # 键值头数量，默认为 32
    embedding_dim: int = 4096  # 嵌入层维度，默认为 4096
    mlp_hidden_dim: int = 11008 # MLP 层隐藏层维度，默认为 11008
    max_position_embeddings: int = 4096  # 最大位置嵌入长度，默认为 4096

    @staticmethod
    def get_name():  # 获取模型名称
        return "meta-llama/Llama-2-7b-hf"  # 返回模型名称

@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):  # 定义 Llama 2 70B 模型的配置类，继承自 Llama2ModelConfig
    num_layers: int = 80   # 模型层数，默认为 80
    num_q_heads: int = 64  # 查询头数量，默认为 64
    num_kv_heads: int = 8  # 键值头数量，默认为 8
    embedding_dim: int = 8192  # 嵌入层维度，默认为 8192
    mlp_hidden_dim: int = 28672 # MLP 层隐藏层维度，默认为 28672
    max_position_embeddings: int = 4096  # 最大位置嵌入长度，默认为 4096

    @staticmethod
    def get_name():  # 获取模型名称
        return "meta-llama/Llama-2-70b-hf"  # 返回模型名称

@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):  # 定义 Llama 3 8B 模型的配置类，继承自 Llama2ModelConfig
    num_layers: int = 32   # 模型层数，默认为 32
    num_q_heads: int = 32  # 查询头数量，默认为 32
    num_kv_heads: int = 8  # 键值头数量，默认为 8
    embedding_dim: int = 4096  # 嵌入层维度，默认为 4096
    mlp_hidden_dim: int = 14336 # MLP 层隐藏层维度，默认为 14336
    max_position_embeddings: int = 4096  # 最大位置嵌入长度，默认为 4096
    rope_theta: Optional[float] = 500000  # RoPE 的θ值，默认为 500000
    vocab_size: int = 128256               # 词汇表大小，默认为 128256

    @staticmethod
    def get_name():  # 获取模型名称
        return "meta-llama/Meta-Llama-3-8B"  # 返回模型名称

@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):  # 定义 Llama 3 70B 模型的配置类，继承自 Llama2ModelConfig
    num_layers: int = 80                # 模型层数，默认为 80
    num_q_heads: int = 64               # 查询头数量，默认为 64
    num_kv_heads: int = 8               # 键值头数量，默认为 8
    embedding_dim: int = 8192           # 嵌入层维度，默认为 8192
    mlp_hidden_dim: int = 28672         # MLP 层隐藏层维度，默认为 28672
    max_position_embeddings: int = 8192 # 最大位置嵌入长度，默认为 8192
    rope_theta: Optional[float] = 500000 # RoPE 的θ值，默认为 500000
    vocab_size: int = 128256             # 词汇表大小，默认为 128256

    @staticmethod
    def get_name():  # 获取模型名称
        return "meta-llama/Meta-Llama-3-70B"  # 返回模型名称

@dataclass
class InternLMModelConfig(Llama2ModelConfig):  # 定义 InternLM 模型的配置类，继承自 Llama2ModelConfig
    max_position_embeddings: int = 4096  # 最大位置嵌入长度，默认为 4096
    vocab_size: int = 103168             # 词汇表大小，默认为 103168

@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):  # 定义 InternLM 20B 模型的配置类，继承自 InternLMModelConfig
    num_layers: int = 60  # 模型层数，默认为 60
    num_q_heads: int = 40 # 查询头数量，默认为 40
    num_kv_heads: int = 40 # 键值头数量，默认为 40
    embedding_dim: int = 5120 # 嵌入层维度，默认为 5120
    mlp_hidden_dim: int = 13824 # MLP 层隐藏层维度，默认为 13824

    @staticmethod
    def get_name():  # 获取模型名称
        return "internlm/internlm-20b"  # 返回模型名称

@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):  # 定义 InternLM2 模型的配置类，继承自 Llama2ModelConfig
    max_position_embeddings: int = 32768 # 最大位置嵌入长度，默认为 32768
    vocab_size: int = 92544             # 词汇表大小，默认为 92544

@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):  # 定义 InternLM2 20B 模型的配置类，继承自 InternLM2ModelConfig
    num_layers: int = 48   # 模型层数，默认为 48
    num_q_heads: int = 48  # 查询头数量，默认为 48
    num_kv_heads: int = 8  # 键值头数量，默认为 8
    embedding_dim: int = 6144  # 嵌入层维度，默认为 6144
    mlp_hidden_dim: int = 16384 # MLP 层隐藏层维度，默认为 16384
    rope_theta: Optional[float] = 1000000  # RoPE 的θ值，默认为 1000000

    @staticmethod
    def get_name():  # 获取模型名称
        return "internlm/internlm2-20b"  # 返回模型名称

@dataclass
class Phi2ModelConfig(Llama2ModelConfig):  # 定义 Phi-2 模型的配置类，继承自 Llama2ModelConfig
    num_layers: int = 32                 # 模型层数，默认为 32
    num_q_heads: int = 32                # 查询头数量，默认为 32
    num_kv_heads: int = 32               # 键值头数量，默认为 32
    embedding_dim: int = 2560            # 嵌入层维度，默认为 2560
    mlp_hidden_dim: int = 10240          # MLP 层隐藏层维度，默认为 10240
    max_position_embeddings: int = 2048  # 最大位置嵌入长度，默认为 2048
    use_gated_mlp: bool = False          # 是否使用门控 MLP，默认为 False
    use_bias: bool = True                # 是否使用偏置，默认为 True
    use_qkv_bias: bool = True            # 是否在 QKV 层使用偏置，默认为 True
    activation: ActivationType = ActivationType.GELU # 激活函数类型，默认为 GELU
    norm: NormType = NormType.LAYER_NORM # 规范化层类型，默认为 LayerNorm
    post_attn_norm: bool = False         # 是否在注意力层之后添加规范化层，默认为 False
    vocab_size: int = 51200              # 词汇表大小，默认为 51200
    rope_scaling: Optional[Dict[str, Any]] = None # RoPE 的扩展参数，默认为 None
    rope_theta: Optional[float] = 10000  # RoPE 的θ值，默认为 10000
    partial_rotary_factor: float = 0.4   # 部分旋转因子，默认为 0.4
    no_tensor_parallel: bool = True      # 是否禁用张量并行，默认为 True

    @staticmethod
    def get_name():  # 获取模型名称
        return "microsoft/phi-2"  # 返回模型名称

@dataclass
class QwenModelConfig(Llama2ModelConfig):  # 定义 Qwen 模型的配置类，继承自 Llama2ModelConfig
    use_qkv_bias: bool = True        # 是否在 QKV 层使用偏置，默认为 True
    max_position_embeddings: int = 32768 # 最大位置嵌入长度，默认为 32768
    vocab_size: int = 152064          # 词汇表大小，默认为 152064

    @staticmethod
    def get_name():  # 获取模型名称
        return "Qwen/Qwen-Config"  # 返回模型名称

@dataclass
class Qwen72BModelConfig(QwenModelConfig):  # 定义 Qwen 72B 模型的配置类，继承自 QwenModelConfig
    num_layers: int = 80   # 模型层数，默认为 80
    num_q_heads: int = 64  # 查询头数量，默认为 64
    num_kv_heads: int = 64 # 键值头数量，默认为 64
    embedding_dim: int = 8192  # 嵌入层维度，默认为 8192
    mlp_hidden_dim: int = 24576 # MLP 层隐藏层维度，默认为 24576
    rope_theta: Optional[float] = 1000000  # RoPE 的θ值，默认为 1000000

    @staticmethod
    def get_name():  # 获取模型名称
        return "Qwen/Qwen-72B"  # 返回模型名称