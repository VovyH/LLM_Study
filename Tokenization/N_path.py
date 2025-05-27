# import re # 未使用，可以移除
from collections import defaultdict # 导入defaultdict，用于创建默认值的字典
# from collections import deque # 未使用，可以移除
from math import log # 导入log函数，用于计算IDF

class Tokenizer: # 定义分词器类
    def __init__(self, dictionary, corpus): # 初始化方法
        self.dictionary = dictionary  # 词典，一个包含所有已知词语的集合
        self.corpus = corpus  # 语料库，一个文档列表，用于计算TF-IDF
        self._build_trie()  # 构建前缀树
        self._compute_tfidf() # 计算词典中每个词的TF-IDF值

    def _build_trie(self): # 私有方法，构建前缀树
        """构建前缀树"""
        self.trie = {} # 初始化空的前缀树（字典实现）
        for word in self.dictionary: #遍历词典中的每一个词
            node = self.trie # 从根节点开始
            for char in word: # 遍历词中的每一个字
                if char not in node: # 如果当前字符不在当前节点的子节点中
                    node[char] = {} # 为当前字符创建一个新的子节点
                node = node[char] # 移动到子节点
            node['#'] = True  # 使用 '#' 标记词语的结束

    def _compute_tfidf(self): # 私有方法，计算TF-IDF值
        """计算TF-IDF值"""
        self.tfidf = {} # 初始化一个空字典来存储TF-IDF值
        doc_count = len(self.corpus) # 语料库中的文档总数
        word_doc_counts = defaultdict(int) # 存储每个词出现在多少个文档中
        for doc in self.corpus: # 遍历语料库中的每个文档
            seen_in_doc = set() # 跟踪本文档中已计数的词，避免重复计数
            for word in self.dictionary: # 遍历词典中的每个词
                if word in doc and word not in seen_in_doc: # 如果词在文档中且尚未计数
                    word_doc_counts[word] += 1 # 该词的文档计数加1
                    seen_in_doc.add(word) # 标记为已在本文件中计数

        for word in self.dictionary: # 遍历词典中的每个词
            # 计算词频 (TF) - 这里简化为词在文档集合中的出现频率（文档频率）
            # 更标准的TF是词在一个文档中的频率，但这里用于词权重，文档频率更接近IDF的用途
            # 或者理解为：包含该词的文档比例
            tf_ratio = word_doc_counts[word] / doc_count if doc_count > 0 else 0 # 包含该词的文档比例
            
            # 计算逆文档频率 (IDF)
            # 使用 log((doc_count + 1) / (word_doc_counts[word] + 1)) + 1 平滑处理，避免除零和取对数问题
            idf = log((doc_count + 1) / (word_doc_counts[word] + 1)) + 1 # 加1确保IDF值为正
            
            self.tfidf[word] = tf_ratio * idf # TF-IDF = TF * IDF (此处TF更像是文档频率比例)
            # 如果希望TF是词在整个语料库的总次数，则需要不同计算方式
            # 当前实现中，tf_ratio * idf 更像是 DF_ratio * IDF，作为词的全局重要性度量

    def _search(self, sentence): # 私有方法，在前缀树中搜索所有可能的词
        """在前缀树中搜索所有可能的词"""
        results = [] # 初始化结果列表，存储找到的词及其位置
        n = len(sentence) # 输入句子的长度
        for i in range(n): # 遍历句子中的每个字符作为潜在词的开始
            node = self.trie # 从前缀树的根节点开始搜索
            for j in range(i, n): # 从当前开始位置i向后遍历
                if sentence[j] not in node: # 如果当前字符不在前缀树的当前节点
                    break # 中断内部循环，当前路径无法构成词典中的词
                node = node[sentence[j]] # 移动到下一个节点
                if '#' in node: # 如果当前节点标记为一个词的结束
                    results.append((i, j, sentence[i:j+1])) # 将(起始索引, 结束索引, 词)存入结果
        return results # 返回所有找到的词及其位置信息

    def n_shortest_paths(self, sentence, N=5): # 找到N条TF-IDF权重最高的路径（“最短”在此处指权重最高）
        """找到N条权重最高的路径 (基于TF-IDF总和)"""
        adj = defaultdict(list) #邻接表，adj[start_idx] 存储 (end_idx, word, weight)
        possible_words = self._search(sentence) # 获取句子中所有可能的词语片段
        # possible_words 是一个列表，包含元组 (start_idx, end_idx_inclusive, word_string)
        
        for i, j, word in possible_words: # 遍历所有可能的词
            # adj的key是词的起始索引，value是列表，包含(词的结束索引+1, 词本身, 词的TF-IDF权重)
            adj[i].append((j + 1, word, self.tfidf.get(word, 0))) 

        completed_paths = [] # 用于存储所有完整的分词路径及其权重
        # 使用DFS（深度优先搜索）来查找所有路径
        # stack中的每个元素是元组: (当前路径分段列表, 当前路径的结束索引)
        # 当前路径分段列表是 [(word1, weight1), (word2, weight2), ...]
        stack = [] 

        # 初始化栈，加入所有以句子开头 (索引0) 的词作为起始路径
        if 0 in adj: # 检查是否存在从索引0开始的词
            for end_idx, word, weight in adj[0]: # 遍历从索引0开始的每个词
                stack.append(([(word, weight)], end_idx)) # 将 (路径分段, 结束索引) 推入栈

        while stack: # 当栈不为空时，继续搜索
            current_path_segments, current_end_idx = stack.pop() # 弹出一个路径进行扩展

            if current_end_idx == len(sentence): # 如果当前路径的结束索引等于句子长度
                completed_paths.append(current_path_segments) # 说明找到了一条完整的分词路径
                continue # 继续处理栈中的其他路径

            # 如果路径未结束，尝试扩展当前路径
            if current_end_idx in adj: # 检查是否存在从current_end_idx开始的词
                for next_end_idx, next_word, next_weight in adj[current_end_idx]: # 遍历每个可能的下一个词
                    # 创建新的路径分段列表
                    new_path_segments = current_path_segments + [(next_word, next_weight)]
                    stack.append((new_path_segments, next_end_idx)) # 将扩展后的路径推入栈

        # 根据路径的总TF-IDF权重对所有完整路径进行排序（降序，权重越高越好）
        completed_paths.sort(key=lambda p: sum(weight for _, weight in p), reverse=True)
        
        return completed_paths[:N] # 返回权重最高的N条路径

    def tokenize(self, sentence, N=5): # 公开的分词方法
        """分词并返回N条最佳路径的字符串表示及其总权重"""
        paths = self.n_shortest_paths(sentence, N) # 获取N条最佳路径（路径是(word, weight)元组的列表）
        
        results_with_weights = []
        for path in paths: # path is like [(word1, weight1), (word2, weight2), ...]
            path_str = '/'.join([word for word, _ in path])
            total_weight = sum(weight for _, weight in path) # 计算路径的总权重
            results_with_weights.append((path_str, total_weight))
        return results_with_weights # 返回 (路径字符串, 总权重) 的列表

# 示例词典
dictionary = {'商务处', '女干事', '商务', '处女', '干事'}

# 示例语料库
corpus = [
    "商务处的职责是处理商务相关事务",
    "女干事在商务处工作",
    "商务谈判需要处女干事参与",
    "干事们在商务处开会"
]

# 示例句子
sentence = "商务处女干事"

# 创建分词器
tokenizer = Tokenizer(dictionary, corpus)

# 分词并获取N条最佳路径
tokenized_results = tokenizer.tokenize(sentence, N=5) # N可以根据需要调整

# 打印N条最佳路径及其总权重，每条路径占一行
if tokenized_results: # 如果找到了分词结果
    print(f"找到的最佳分词路径 (N={len(tokenized_results)}):")
    for i, (path_str, weight) in enumerate(tokenized_results): # 遍历每条路径字符串及其权重
        print(f"{i+1}. {path_str} (总权重: {weight:.4f})") # 打印路径和总权重，权重保留4位小数
else: # 如果没有找到有效的分词路径
    print("未能找到有效的分词路径。")
