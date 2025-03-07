from enum import Enum  # 从enum模块导入Enum类
from typing import Union  # 从typing模块导入Union类型

from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
    FlashinferAttentionWrapper,  # 从指定路径导入FlashinferAttentionWrapper类
)
from sarathi.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,  # 从指定路径导入NoOpAttentionWrapper类
)
from sarathi.types import AttentionBackend  # 从sarathi.types模块导入AttentionBackend枚举类

ATTENTION_BACKEND = AttentionBackend.NO_OP  # 初始化全局变量ATTENTION_BACKEND为NO_OP

def set_attention_backend(backend: Union[str, AttentionBackend]):  # 定义设置注意力后端的函数，接受字符串或AttentionBackend类型的参数
    if isinstance(backend, str):  # 如果backend是字符串类型
        backend = backend.upper()  # 将backend转换为大写
        if backend not in AttentionBackend.__members__:  # 如果转换后的backend不在AttentionBackend的成员中
            raise ValueError(f"Unsupported attention backend: {backend}")  # 抛出不支持的注意力后端错误
        backend = AttentionBackend[backend]  # 将字符串转换为对应的AttentionBackend枚举成员
    elif not isinstance(backend, AttentionBackend):  # 否则，如果backend不是AttentionBackend类型
        raise ValueError(f"Unsupported attention backend: {backend}")  # 抛出不支持的注意力后端错误

    global ATTENTION_BACKEND  # 声明使用全局变量ATTENTION_BACKEND
    ATTENTION_BACKEND = backend  # 设置全局变量ATTENTION_BACKEND为指定的backend

def get_attention_wrapper():  # 定义获取注意力包装器的函数
    if ATTENTION_BACKEND == AttentionBackend.FLASHINFER:  # 如果当前ATTENTION_BACKEND是FLASHINFER
        return FlashinferAttentionWrapper.get_instance()  # 返回FlashinferAttentionWrapper的单例实例
    elif ATTENTION_BACKEND == AttentionBackend.NO_OP:  # 否则，如果ATTENTION_BACKEND是NO_OP
        return NoOpAttentionWrapper.get_instance()  # 返回NoOpAttentionWrapper的单例实例

    raise ValueError(f"Unsupported attention backend: {ATTENTION_BACKEND}")  # 如果不支持的后端，抛出错误
