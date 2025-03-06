from abc import ABC  # 导入抽象基类模块，用于定义抽象类
from dataclasses import dataclass  # 导入数据类装饰器，用于简化类的定义
from typing import Any  # 导入Any类型，表示可以是任意类型

from vidur.config.utils import get_all_subclasses  # 导入工具函数，用于获取某个类的所有子类


@dataclass  # 使用dataclass装饰器，自动生成__init__等方法
class BaseFixedConfig(ABC):  # 定义一个抽象基类，继承自ABC，表示这是一个抽象类

    @classmethod  # 定义类方法，可以通过类名直接调用
    def create_from_type(cls, type_: Any) -> Any:  # 根据类型创建子类实例的方法
        for subclass in get_all_subclasses(cls):  # 遍历当前类的所有子类
            if subclass.get_type() == type_:  # 如果子类的get_type方法返回值等于传入的type_
                return subclass()  # 返回该子类的实例
        raise ValueError(f"[{cls.__name__}] Invalid type: {type_}")  # 如果没有匹配的子类，抛出异常

    @classmethod  # 定义类方法，可以通过类名直接调用
    def create_from_name(cls, name: str) -> Any:  # 根据名称创建子类实例的方法
        for subclass in get_all_subclasses(cls):  # 遍历当前类的所有子类
            if subclass.get_name() == name:  # 如果子类的get_name方法返回值等于传入的name
                return subclass()  # 返回该子类的实例
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")  # 如果没有匹配的子类，抛出异常

    @classmethod  # 定义类方法，可以通过类名直接调用
    def create_from_type_string(cls, type_str: str) -> Any:  # 根据类型字符串创建子类实例的方法
        for subclass in get_all_subclasses(cls):  # 遍历当前类的所有子类
            if str(subclass.get_type()) == type_str:  # 如果子类的get_type方法返回值转为字符串后等于传入的type_str
                return subclass()  # 返回该子类的实例
        raise ValueError(f"[{cls.__name__}] Invalid type string: {type_str}")  # 如果没有匹配的子类，抛出异常
