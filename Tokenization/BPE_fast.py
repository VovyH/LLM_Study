import time
from collections import defaultdict

def get_freq(word_freq):
    bigram_frequency = defaultdict(int)
    for word, freq in word_freq.items():
        unigrams = word.split(" ")
        for i in range(len(unigrams) - 1):
            bigram = (unigrams[i], unigrams[i+1])
            bigram_frequency[bigram] += freq
    return bigram_frequency

def argmax(bigram_frequency):
    return max(bigram_frequency.items(), key=lambda x: x[1])[0]

def merge_bigram(best_bigram, new_unigram, word_freq):
    new_word_freq = {}
    for word, freq in word_freq.items():
        unigrams = word.split(" ")
        i = 0
        new_unigrams = []
        while i < len(unigrams):
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

# ----------- 加速版BPE -----------
class FastBPE:
    def __init__(self, word_freq):
        self.word_freq = word_freq.copy()
        self.words = [word.split(" ") for word in word_freq]
        self.freqs = list(word_freq.values())
        self.active_bigrams = defaultdict(int)
        self.bigram_pos = defaultdict(list)  # 记录bigram出现在哪些词的哪些位置

    def init_bigrams(self):
        for idx, word in enumerate(self.words):
            for i in range(len(word)-1):
                bigram = (word[i], word[i+1])
                self.active_bigrams[bigram] += self.freqs[idx]
                self.bigram_pos[bigram].append((idx, i))

    def update_bigrams(self, best_bigram, new_unigram):
        # 只更新受影响的bigram
        positions = self.bigram_pos[best_bigram]
        self.bigram_pos.pop(best_bigram)
        for idx, i in positions:
            word = self.words[idx]
            if i >= len(word)-1 or (word[i], word[i+1]) != best_bigram:
                continue
            # 合并
            word = word[:i] + [new_unigram] + word[i+2:]
            self.words[idx] = word
            # 更新bigram统计
            # 先移除旧bigram
            if i > 0:
                left = (word[i-1], new_unigram)
                self.active_bigrams[left] += self.freqs[idx]
                self.bigram_pos[left].append((idx, i-1))
            if i < len(word)-1:
                right = (new_unigram, word[i+1])
                self.active_bigrams[right] += self.freqs[idx]
                self.bigram_pos[right].append((idx, i))
        # 清空已合并bigram的频率
        self.active_bigrams[best_bigram] = 0

    def run(self, target_vocab_size):
        vocab = set()
        merge_rule = []
        for word in self.words:
            for unigram in word:
                vocab.add(unigram)
        vocab = list(vocab)
        self.init_bigrams()
        while len(vocab) < target_vocab_size:
            if not self.active_bigrams:
                break
            best_bigram = max(self.active_bigrams.items(), key=lambda x: x[1])[0]
            if self.active_bigrams[best_bigram] == 0:
                break
            new_unigram = "".join(best_bigram)
            merge_rule.append((best_bigram, new_unigram))
            vocab.append(new_unigram)
            self.update_bigrams(best_bigram, new_unigram)
        return vocab, merge_rule

if __name__ == "__main__":
    import random

    # 构造较大的中文词频表
    base_words = [
        "我 爱 北 京", "我 爱 上 海", "你 爱 北 京", "他 去 上 海",
        "她 爱 北 京", "我们 去 上 海", "他们 爱 北 京", "你们 去 上 海"
    ]
    word_freq = {}
    for w in base_words:
        word_freq[w] = random.randint(10, 100)
    target_vocab_size = 30

    # 原始BPE计时
    t1 = time.time()
    vocab1, merge_rule1 = BPE(word_freq, target_vocab_size)
    t2 = time.time()
    print("原始BPE耗时：%.6f秒" % (t2-t1))

    # 加速BPE计时
    t3 = time.time()
    fast_bpe = FastBPE(word_freq)
    vocab2, merge_rule2 = fast_bpe.run(target_vocab_size)
    t4 = time.time()
    print("加速BPE耗时：%.6f秒" % (t4-t3))

    print("原始BPE词表：", vocab1)
    print("加速BPE词表：", vocab2)
    print("原始BPE合并规则：", merge_rule1)
    print("加速BPE合并规则：", merge_rule2)