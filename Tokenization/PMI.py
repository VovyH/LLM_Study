import math
from collections import Counter, defaultdict

def compute_pmi(corpus, vocab):
    # 统计unigram和bigram频率
    unigram_counts = Counter()
    bigram_counts = Counter()
    total_unigrams = 0
    for word in corpus:
        chars = list(word)
        unigram_counts.update(chars)
        total_unigrams += len(chars)
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i+1]
            bigram_counts[bigram] += 1
    # 计算PMI
    pmi_scores = {}
    for bigram, bigram_count in bigram_counts.items():
        a, b = bigram[0], bigram[1]
        p_a = unigram_counts[a] / total_unigrams
        p_b = unigram_counts[b] / total_unigrams
        p_ab = bigram_count / (total_unigrams - 1)
        pmi = math.log(p_ab / (p_a * p_b) + 1e-10)  # 防止除零
        pmi_scores[bigram] = pmi
    return pmi_scores

def train_wordpiece(corpus, vocab_size):
    # 初始化词表为所有字符
    vocab = set()
    for word in corpus:
        vocab.update(list(word))
    vocab = set(vocab)
    # 词表合并
    while len(vocab) < vocab_size:
        pmi_scores = compute_pmi(corpus, vocab)
        if not pmi_scores:
            break
        # 找到PMI最大的bigram
        best_bigram = max(pmi_scores, key=pmi_scores.get)
        vocab.add(best_bigram)
        # 用新bigram替换corpus中的字符对
        new_corpus = []
        for word in corpus:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] + word[i+1] == best_bigram:
                    new_word.append(best_bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(new_word)
        corpus = ["".join(w) for w in new_corpus]
    return vocab

# 示例用法
if __name__ == "__main__":
    corpus = ["中国人", "中国人民", "爱中国", "人民币"]
    vocab = train_wordpiece(corpus, vocab_size=15)
    print("训练得到的词表:", vocab)