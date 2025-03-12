from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler  # 导入LOR全局调度器类
from vidur.scheduler.global_scheduler.random_global_scheduler import (  # 导入随机全局调度器类
    RandomGlobalScheduler,
)
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (  # 导入轮询全局调度器类
    RoundRobinGlobalScheduler,
)
from vidur.types import GlobalSchedulerType  # 导入全局调度器类型枚举类
from vidur.utils.base_registry import BaseRegistry  # 导入基础注册表类，用于实现注册机制


class GlobalSchedulerRegistry(BaseRegistry):  # 定义全局调度器注册表类，继承自BaseRegistry
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:  # 定义类方法，将字符串转换为全局调度器类型枚举值
        return GlobalSchedulerType.from_str(key_str)  # 调用GlobalSchedulerType的from_str方法进行转换


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)  # 注册随机全局调度器到注册表中
GlobalSchedulerRegistry.register(  # 注册轮询全局调度器到注册表中
    GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)  # 注册LOR全局调度器到注册表中
