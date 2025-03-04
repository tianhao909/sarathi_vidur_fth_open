""" 
Singleton metaclass as described in
https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
"""

class Singleton(type):  # 定义一个元类，继承自type，用于实现单例模式
    _instances = {}  # 定义一个类变量，用于存储每个类的唯一实例

    def __call__(cls, *args, **kwargs):  # 元类的__call__方法会在每次创建类实例时被调用
        if cls not in cls._instances:  # 检查当前类是否已经在_instances字典中
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)  # 如果不在，则调用父类的__call__方法创建实例并存储
        return cls._instances[cls]  # 返回存储在_instances中的实例，确保始终返回同一个实例
