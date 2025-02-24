class AttentionInput:  # 定义一个名为 AttentionInput 的类
    def __init__(  # 定义类的构造函数
        self,  # 类实例自身
        prefill_chunk_size: int,  # 预填充块大小，类型为整数
        kv_cache_size: int,  # KV 缓存大小，类型为整数
        batch_size: int,  # 批量大小，类型为整数
        is_prefill: bool,  # 是否为预填充，类型为布尔值
    ):
        self.prefill_chunk_size = prefill_chunk_size  # 初始化 prefill_chunk_size 属性
        self.kv_cache_size = kv_cache_size  # 初始化 kv_cache_size 属性
        self.batch_size = batch_size  # 初始化 batch_size 属性
        self.is_prefill = is_prefill  # 初始化 is_prefill 属性

    def is_valid(self, max_seq_len: int):  # 定义一个方法用于验证输入是否合法，参数为最大序列长度
        if self.is_prefill:  # 如果当前是预填充模式
            if self.batch_size != 1:  # 检查批量大小是否不等于1
                return False  # 如果不是，返回 False
            elif self.prefill_chunk_size == 0:  # 检查预填充块大小是否为0
                return False  # 如果是，返回 False
            elif self.prefill_chunk_size + self.kv_cache_size > max_seq_len:  # 检查预填充块大小加上 KV 缓存大小是否超过最大序列长度
                return False  # 如果超过，返回 False
        else:  # 如果不是预填充模式
            if self.prefill_chunk_size > 0:  # 检查预填充块大小是否大于0
                return False  # 如果是，返回 False
            elif self.kv_cache_size == 0:  # 检查 KV 缓存大小是否为0
                return False  # 如果是，返回 False
            elif self.kv_cache_size > max_seq_len:  # 检查 KV 缓存大小是否超过最大序列长度
                return False  # 如果超过，返回 False
        return True  # 如果所有条件都通过，返回 True

    def is_under_memory_limit(self, max_num_tokens: int):  # 定义一个方法用于检查是否在内存限制之下，参数为最大令牌数
        return (  # 返回以下计算结果
            self.batch_size * (self.kv_cache_size + self.prefill_chunk_size)  # 计算批量大小乘以 (KV 缓存大小加预填充块大小)
            <= max_num_tokens  # 检查是否小于或等于最大令牌数
        )

    def __str__(self):  # 定义类的字符串表示方法
        return f"prefill_chunk_size: {self.prefill_chunk_size}, kv_cache_size: {self.kv_cache_size}, batch_size: {self.batch_size}, is_prefill: {self.is_prefill}"  # 返回包含所有属性值的格式化字符串