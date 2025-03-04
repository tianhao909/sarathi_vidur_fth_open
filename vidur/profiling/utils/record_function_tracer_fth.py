import json  # 导入用于处理JSON数据的模块
import uuid  # 导入用于生成唯一标识符的模块

import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch库，用于深度学习框架


class RecordFunctionTracer:  # 定义一个名为RecordFunctionTracer的类
    def __init__(self, output_path: str):  # 类的构造函数，接受输出路径作为参数
        trace_id = str(uuid.uuid4())[:8]  # 生成一个8位的唯一追踪ID
        self.trace_path = (  # 设置追踪文件的路径
            f"{output_path}/profiler_traces/profiler_trace_{trace_id}.json"  # 使用f-string格式化路径字符串
        )

    def __enter__(self):  # 定义上下文管理器的进入方法
        self.profiler = torch.profiler.profile(  # 创建一个PyTorch的Profiler实例
            activities=[  # 指定需要跟踪的活动
                torch.profiler.ProfilerActivity.CPU,  # 跟踪CPU活动
                torch.profiler.ProfilerActivity.CUDA,  # 跟踪CUDA（GPU）活动
            ],
        )
        self.profiler.__enter__()  # 进入Profiler上下文

    def __exit__(self, *args):  # 定义上下文管理器的退出方法
        self.profiler.__exit__(None, None, None)  # 退出Profiler上下文
        torch.cuda.synchronize()  # 同步CUDA，确保所有GPU操作完成
        self.profiler.export_chrome_trace(self.trace_path)  # 将追踪结果导出为Chrome可视化格式的文件

    def find_children(self, trace, event):  # 定义一个方法，用于查找事件的子事件
        if not ("dur" in event and "ts" in event):  # 检查事件是否包含'dur'（持续时间）和'ts'（时间戳）字段
            return  # 如果不包含，返回None

        children = []  # 初始化一个空列表，用于存储子事件
        for e in trace:  # 遍历整个追踪列表
            if not ("dur" in e and "ts" in e):  # 检查子事件是否包含'dur'和'ts'字段
                continue  # 如果不包含，跳过该事件

            # 如果子事件的时间戳完全包含在父事件的时间戳内
            if (
                e["ts"] > event["ts"]  # 子事件的开始时间在父事件开始时间之后
                and e["ts"] + e["dur"] < event["ts"] + event["dur"]  # 子事件的结束时间在父事件结束时间之前
            ):
                children.append(e)  # 将符合条件的子事件添加到列表中
        return children  # 返回子事件列表

    def find_correlated_event(self, trace, event):  # 定义一个方法，用于查找与给定事件相关联的事件
        if not ("args" in event and "correlation" in event["args"]):  # 检查事件是否包含'args'和'correlation'字段
            return  # 如果不包含，返回None

        for e in trace:  # 遍历整个追踪列表
            if not ("args" in e and "correlation" in e["args"]):  # 检查子事件是否包含'args'和'correlation'字段
                continue  # 如果不包含，跳过该事件

            if e == event:  # 如果子事件和目标事件相同
                continue  # 跳过该事件

            if e["args"]["correlation"] == event["args"]["correlation"]:  # 如果子事件的'correlation'值与目标事件相同
                return e  # 返回该相关事件

    def get_operation_time_stats(self):  # 定义一个方法，用于获取操作时间的统计数据
        stats = {}  # 初始化一个空字典，用于存储统计结果

        trace = json.load(open(self.trace_path, "r"))["traceEvents"]  # 加载并读取追踪JSON文件中的'traceEvents'数据

        for event in trace:  # 遍历每一个事件
            if not ("cat" in event and event["cat"] == "user_annotation"):  # 检查事件是否属于'user_annotation'类别
                continue  # 如果不属于，跳过该事件
            children = self.find_children(trace, event)  # 查找该事件的子事件
            cuda_time = 0  # 初始化CUDA时间为0
            for child in children:  # 遍历所有子事件
                if not ("cat" in child and child["cat"] == "cuda_runtime"):  # 检查子事件是否属于'cuda_runtime'类别
                    continue  # 如果不属于，跳过该事件
                correlated_event = self.find_correlated_event(trace, child)  # 查找与子事件相关联的事件
                if not correlated_event:  # 如果没有找到相关事件
                    continue  # 跳过该子事件
                cuda_time += correlated_event["dur"]  # 将相关事件的持续时间累加到cuda_time
            if cuda_time == 0:  # 如果CUDA时间为0
                continue  # 跳过该事件

            name = event["name"].replace("vidur_", "")  # 获取事件名称，并移除前缀"vidur_"

            if name not in stats:  # 如果该名称尚未在统计字典中
                stats[name] = []  # 初始化一个空列表以存储时间数据

            stats[name].append(cuda_time * 1e-3)  # 将CUDA时间转换为毫秒并添加到对应名称的列表中

        return {  # 返回一个包含统计数据的字典
            operation: {  # 对于每个操作
                "min": np.min(times),  # 计算最小时间
                "max": np.max(times),  # 计算最大时间
                "mean": np.mean(times),  # 计算平均时间
                "median": np.median(times),  # 计算中位数时间
                "std": np.std(times),  # 计算时间的标准差
            }
            for operation, times in stats.items()  # 对统计字典中的每个操作及其时间列表进行迭代
        }
