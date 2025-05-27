# 子索引向左不断缩小，父索引向右靠，所以是FMM
def forward_max_matching(sentence, dictionary, max_len=6):
    result = []  # 初始化结果列表，用于存储分词结果
    i = 0  # 初始化指针i为0，表示从句子的开头开始匹配
    while i < len(sentence):  # 遍历句子直到结束
        matched = False  # 初始化匹配标志为False
        for j in range(max_len, 0, -1):  # 从最大长度6 到最小长度0 尝试匹配
            if i + j > len(sentence):  # 如果当前位置加上长度j超出句子长度，跳过
                continue
            word = sentence[i:i+j]  # 提取从i到i+j的子串
            if word in dictionary:  # 如果子串在词典中
                result.append(word)  # 将匹配的词语添加到结果列表
                i += j  # 更新指针i的位置
                matched = True  # 标记为已找到匹配
                break  # 跳出内层循环
            
        if not matched:  # 遍历完子串，如果没有找到匹配
            result.append(sentence[i])  # 将单个字符作为词语添加到结果列表
            i += 1  # 更新指针i的位置
    return result  # 返回分词结果

# 示例词典
dictionary = {'商务处', '女干事', '商务', '处女', '干事'}  # 包含所有可能的词

# 示例句子
sentence = "商务处女干事是处女"

# 分词：商务处/女干事/是/处女
print('/'.join(forward_max_matching(sentence, dictionary)))