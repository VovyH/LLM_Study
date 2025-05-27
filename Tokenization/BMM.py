# 子索引向左不断缩小，父索引向左，取（i-j,i）,j不断缩小所以子串也会缩小
def backward_max_matching(sentence, dictionary, max_len=6):
    result = []  # 初始化结果列表，用于存储分词结果
    i = len(sentence)  # 初始化指针i为句子的长度，表示从句子的末尾开始匹配
    while i > 0:  # 遍历句子直到开头
        matched = False  # 初始化匹配标志为False
        for j in range(max_len, 0, -1):  # 从最大长度到最小长度尝试匹配
            if i - j < 0:  # 如果当前位置减去长度j小于0，跳过
                continue
            word = sentence[i-j:i]  # 提取从i-j到i的子串
            if word in dictionary:  # 如果子串在词典中
                result.insert(0, word)  # 将匹配的词语添加到结果列表的开头
                i -= j  # 更新指针i的位置
                matched = True  # 标记为已找到匹配
                break  # 跳出内层循环
            
        if not matched:  # 如果没有找到匹配
            result.insert(0, sentence[i-1])  # 将单个字符作为词语添加到结果列表的开头
            i -= 1  # 更新指针i的位置
    return result  # 返回分词结果

# 示例词典
dictionary = {'商务处', '女干事', '商务', '处女', '干事'}  # 包含所有可能的词

# 示例句子
sentence = "商务处女干事是处女"

# 分词：商务处/女干事/是/处女
print('/'.join(backward_max_matching(sentence, dictionary))) 