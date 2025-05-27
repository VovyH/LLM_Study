def BPE():
    while len(vocab) < target_vocab_size:
        # 统计新词表导致的bigram频率
        bigram_frequency = get_freq(word_freq)
        
        # 找到频率最大的bigram
        best_bigram = argmax(bigram_frequency)
        
        # 新词为频率最大的bigram的连接
        new_unigram = ''.join(best_bigram)
        
        # 对词频表中每个词更新其切词方案(合并best_bigram, new_unigram)
        word_freq = merge_bigram(best_bigram, new_unigram, word_freq)
        
        # 添加合并规则、添加新词
        merge_rule.append((best_bigram, new_unigram))
        vocab.append(new_unigram)

def get_freq(word_freq):
    bigram_frequency = {}
    # 对于word_freq中每个词word和对应的频率freq
    for word, freq in word_freq.items():
        # 将word按当前切词方案切开
        unigrams = word.split("/")
        for unigram in unigrams:
            # 统计bigram频率
            bigram_frequency[(unigram, next_unigram)] += freq
    return bigram_frequency

def merge_bigram(best_bigram, new_unigram, word_freq):
    # 对于带切词信息的词频表word_freq中的每个词
    for word in word_freq:
        # 如果里面有best_bigram，合成 new_unigram
        word = word.replace(best_bigram, new_unigram)
    return word_freq
