import torch
from labml_nn.sampling import Sampler

class TopKSampler(Sampler):
    def __init__(self, k: int, sampler: Sampler):
        self.k = k
        self.sampler = Sampler
        
    def __call__(self, logits: torch.Tensor):
        zeros = logits.new_ones(logits.shape) * float('-inf')
        """
        所有生成的：logits维度为(batchsize, seq_len, vocab_size)
        每生成一个token的维度：(batchsize, vocab_size)
        topk(): 取得最后一个维度张量即(vocab_size)表示每个词汇的概率，选择最大的k个
        """
        values, indices = torch.topk(logits, self.k, dim=-1)
        # 将value插入到zeros最后一个维度上
        zeros.scatter_(-1, indices, values)
        return self.sampler(zeros)
        