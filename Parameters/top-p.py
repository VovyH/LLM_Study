import torch
from labml_nn.sampling import Sampler

class TopPSampler(Sampler):
    def __init__(self, p: float, sampler: Sampler):
        self.p = p
        self.sampler = sampler
        # Softmax(dim=-1): 对tensor的最后一个维度，即vocab_size
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def __call__(self, logits: torch.Tensor):
        # 1.归一化最后一个维度vocab的概率
        probs = self.softmax(logits)
        
        # 2.进行排序，descending为True表示从大到小排序
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        
        # 3.累加
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 4.找出累积概率小于p的位置
        nucleus = cum_sum_probs < self.p
        
        # 5.确保至少包含一个词元
        nucleus = torch.cat([
            nucleus.new_ones(nucleus.shape[:-1] + (1,)), 
            nucleus[..., :-1]
        ], dim=-1)
        
        # 6.应用掩码
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')
        
        # 7.采样并映射回原始索引空间
        sampled_sorted_indexes = self.sampler(sorted_log_probs)
        
        # 确保索引张量的维度与sorted_indices匹配
        # 这里需要使用gather从sorted_indices中收集原始索引
        res = torch.gather(
            sorted_indices,
            dim=-1,
            index=sampled_sorted_indexes
        )
        
        return res

# 示例Sampler实现
class LogitsSampler(Sampler):
    def __call__(self, logits: torch.Tensor):
        # 简单实现：从logits中采样最大概率的索引
        # 确保返回的索引形状为 [batch_size, seq_len, 1]
        return torch.argmax(logits, dim=-1, keepdim=True)

def main():
    # 创建一个示例logits张量 (batch_size=2, seq_len=1, vocab_size=10)
    logits = torch.tensor([
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ],
        [
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        ]
    ])
    
    # 创建基础采样器
    base_sampler = LogitsSampler()
    
    # 创建Top-P采样器 (p=0.9)
    top_p_sampler = TopPSampler(p=0.9, sampler=base_sampler)
    
    # 执行采样
    sampled_indices = top_p_sampler(logits)
    
    print("原始logits:")
    print(logits)
    
    print("\n采样结果 (索引):")
    print(sampled_indices)
    
    # 验证结果
    for i in range(logits.shape[0]):
        # 获取采样的索引
        idx = sampled_indices[i, 0, 0].item()
        
        # 获取对应的概率
        probs = torch.softmax(logits[i, 0], dim=-1)
        print(f"\n样本 {i}:")
        print(f"采样索引: {idx}")
        print(f"对应概率: {probs[idx]:.4f}")
        print(f"累积概率截止: {torch.sum(probs[probs >= probs[idx]]):.4f}")

if __name__ == "__main__":
    main()