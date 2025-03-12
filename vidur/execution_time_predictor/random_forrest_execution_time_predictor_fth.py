from sklearn.ensemble import RandomForestRegressor  # 从 sklearn.ensemble 模块导入 RandomForestRegressor 类

from vidur.config import (  # 从 vidur.config 模块导入相关配置类
    BaseReplicaSchedulerConfig,  # 导入 BaseReplicaSchedulerConfig 类
    MetricsConfig,  # 导入 MetricsConfig 类
    RandomForrestExecutionTimePredictorConfig,  # 导入 RandomForrestExecutionTimePredictorConfig 类
    ReplicaConfig,  # 导入 ReplicaConfig 类
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (  # 从 vidur.execution_time_predictor.sklearn_execution_time_predictor 模块导入 SklearnExecutionTimePredictor 类
    SklearnExecutionTimePredictor,  # 导入 SklearnExecutionTimePredictor 类
)


class RandomForrestExecutionTimePredictor(SklearnExecutionTimePredictor):  # 定义 RandomForrestExecutionTimePredictor 类，继承自 SklearnExecutionTimePredictor
    def __init__(  # 定义初始化方法
        self,  # 类的实例
        predictor_config: RandomForrestExecutionTimePredictorConfig,  # 预测器配置参数，类型为 RandomForrestExecutionTimePredictorConfig
        replica_config: ReplicaConfig,  # 副本配置参数，类型为 ReplicaConfig
        replica_scheduler_config: BaseReplicaSchedulerConfig,  # 副本调度器配置参数，类型为 BaseReplicaSchedulerConfig
        metrics_config: MetricsConfig,  # 评估指标配置参数，类型为 MetricsConfig
    ) -> None:  # 初始化方法不返回任何值
        # will trigger model training  # 将触发模型训练
        super().__init__(  # 调用父类（SklearnExecutionTimePredictor）的初始化方法
            predictor_config=predictor_config,  # 传递 predictor_config 参数
            replica_config=replica_config,  # 传递 replica_config 参数
            replica_scheduler_config=replica_scheduler_config,  # 传递 replica_scheduler_config 参数
            metrics_config=metrics_config,  # 传递 metrics_config 参数
        )

    def _get_grid_search_params(self):  # 定义获取网格搜索参数的方法
        return {  # 返回一个字典，包含网格搜索的参数
            "n_estimators": self._config.num_estimators,  # 设置 n_estimators 参数，值来自配置中的 num_estimators
            "max_depth": self._config.max_depth,  # 设置 max_depth 参数，值来自配置中的 max_depth
            "min_samples_split": self._config.min_samples_split,  # 设置 min_samples_split 参数，值来自配置中的 min_samples_split
        }

    def _get_estimator(self):  # 定义获取估计器的方法
        return RandomForestRegressor()  # 返回一个 RandomForestRegressor 实例
