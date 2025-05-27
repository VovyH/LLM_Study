# BPE分词算法中文示例
from collections import defaultdict

def get_freq(word_freq):
    """
    统计所有词中每个bigram（二元组）的出现频率
    word_freq: 词频字典，key为带空格分隔的词，value为频率
    返回：bigram频率字典
    """
    bigram_frequency = defaultdict(int)
    """
    这里的时间复杂度是：O(NM)，N是word的数量，M是词的平均长度（形似一个二维矩阵）
    关键：每次经过merge_rule后会得到新的vocab，然后计算得到新的word_freq，又得经过同样的时间复杂度（有些已经计算过了）
    """ 
    for word, freq in word_freq.items():
        unigrams = word.split(" ")
        for i in range(len(unigrams) - 1):
            bigram = (unigrams[i], unigrams[i+1])
            bigram_frequency[bigram] += freq
    return bigram_frequency

def argmax(bigram_frequency):
    """
    返回频率最高的bigram
    """
    return max(bigram_frequency.items(), key=lambda x: x[1])[0]

def merge_bigram(best_bigram, new_unigram, word_freq):
    """
    将所有词中的best_bigram合并为new_unigram
    返回新的word_freq
    """
    new_word_freq = {}
    for word, freq in word_freq.items():
        unigrams = word.split(" ")
        i = 0
        new_unigrams = []
        while i < len(unigrams):
            # 检查当前位置和下一个位置是否组成best_bigram
            if i < len(unigrams) - 1 and (unigrams[i], unigrams[i+1]) == best_bigram:
                new_unigrams.append(new_unigram)
                i += 2
            else:
                new_unigrams.append(unigrams[i])
                i += 1
        new_word = " ".join(new_unigrams)
        new_word_freq[new_word] = freq
    return new_word_freq

def BPE(word_freq, target_vocab_size):
    """
    BPE主流程
    word_freq: 初始词频表，key为带空格分隔的unigram，value为频率
    target_vocab_size: 目标词表大小
    返回：最终词表和合并规则
    """
    vocab = set()
    merge_rule = []
    for word in word_freq:
        for unigram in word.split(" "):
            vocab.add(unigram)
    vocab = list(vocab)
    while len(vocab) < target_vocab_size:
        bigram_frequency = get_freq(word_freq)
        if not bigram_frequency:
            break
        best_bigram = argmax(bigram_frequency)
        new_unigram = "".join(best_bigram)
        word_freq = merge_bigram(best_bigram, new_unigram, word_freq)
        merge_rule.append((best_bigram, new_unigram))
        vocab.append(new_unigram)
    return vocab, merge_rule

if __name__ == "__main__":
    # 示例：初始词频表，每个词用空格分隔unigram，value为频率
    word_freq = {
        "我 爱 北 京": 1,
        "我 爱 上 海": 1,
        "你 爱 北 京": 1,
        "他 去 上 海": 1
    }
    target_vocab_size = 12  # 目标词表大小
    vocab, merge_rule = BPE(word_freq, target_vocab_size)
    print("最终词表：", vocab)
    print("合并规则：", merge_rule)

    # 展示每个词应用BPE分词后的最终切分结果
    def apply_bpe(word, merge_rule):
        unigrams = word.split(" ")
        for bigram, new_unigram in merge_rule:
            i = 0
            while i < len(unigrams) - 1:
                if (unigrams[i], unigrams[i+1]) == bigram:
                    unigrams = unigrams[:i] + [new_unigram] + unigrams[i+2:]
                    # 合并后不递增i，继续检查当前位置
                else:
                    i += 1
        return unigrams

    print("每个词BPE分词结果：")
    for word in word_freq:
        result = apply_bpe(word, merge_rule)
        print(f"{word} -> {'/'.join(result)}")