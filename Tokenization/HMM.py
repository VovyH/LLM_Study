# 简单HMM分词训练与测试代码（适合小白，含详细注释）

from collections import defaultdict
import math

class HMM:
    def __init__(self):
        # 状态集合
        self.states = ['B', 'M', 'E', 'S']
        # 统计用的字典
        self.start_p = defaultdict(int)      # 初始状态计数
        self.trans_p = defaultdict(lambda: defaultdict(int))  # 转移计数
        self.emit_p = defaultdict(lambda: defaultdict(int))   # 发射计数
        # 概率
        self.start_prob = {}
        self.trans_prob = {}
        self.emit_prob = {}

    def _get_label(self, word):
        # 根据词长度生成BMES标签
        if len(word) == 1:
            return ['S']
        elif len(word) == 2:
            return ['B', 'E']
        else:
            return ['B'] + ['M'] * (len(word) - 2) + ['E']

    def train(self, sentences):
        # 训练HMM参数
        for sentence in sentences:
            words = sentence.strip().split()
            obs = ''.join(words)
            labels = []
            for word in words:
                labels.extend(self._get_label(word))
            # 统计初始状态
            self.start_p[labels[0]] += 1
            # 统计转移和发射
            for i in range(len(obs)):
                self.emit_p[labels[i]][obs[i]] += 1
                if i > 0:
                    self.trans_p[labels[i-1]][labels[i]] += 1

        # 归一化为概率
        total_start = sum(self.start_p.values())
        self.start_prob = {s: self.start_p[s] / total_start for s in self.states}
        self.trans_prob = {}
        for s1 in self.states:
            total = sum(self.trans_p[s1].values()) or 1
            self.trans_prob[s1] = {s2: self.trans_p[s1][s2] / total for s2 in self.states}
        self.emit_prob = {}
        for s in self.states:
            total = sum(self.emit_p[s].values()) or 1
            self.emit_prob[s] = {c: self.emit_p[s][c] / total for c in self.emit_p[s]}

    def viterbi(self, text):
        # 维特比算法
        V = [{}]
        path = {}
        for s in self.states:
            V[0][s] = self.start_prob.get(s, 1e-8) * self.emit_prob.get(s, {}).get(text[0], 1e-8)
            path[s] = [s]
        for t in range(1, len(text)):
            V.append({})
            new_path = {}
            for curr in self.states:
                prob_state = []
                for prev in self.states:
                    prob = V[t-1][prev] * self.trans_prob.get(prev, {}).get(curr, 1e-8) * \
                           self.emit_prob.get(curr, {}).get(text[t], 1e-8)
                    prob_state.append((prob, prev))
                max_prob, max_state = max(prob_state)
                V[t][curr] = max_prob
                new_path[curr] = path[max_state] + [curr]
            path = new_path
        # 取最后一个字最大概率的路径
        final_probs = [(V[-1][s], s) for s in self.states]
        max_prob, max_state = max(final_probs)
        return path[max_state]

    def cut(self, text):
        # 分词主函数
        state_seq = self.viterbi(text)
        result = []
        word = ''
        for i, char in enumerate(text):
            state = state_seq[i]
            if state == 'B':
                word = char
            elif state == 'M':
                word += char
            elif state == 'E':
                word += char
                result.append(word)
                word = ''
            elif state == 'S':
                result.append(char)
        if word:
            result.append(word)
        return result

if __name__ == "__main__":
    # 捏造一个小语料库（每行是一个分好词的句子，词之间用空格分隔）
    train_data = [
        "我 爱 北京 天安门",
        "你 爱 上海 东方明珠",
        "我 是 学生",
        "你 是 老师",
        "北京 是 中国 的 首都",
        "上海 是 中国 的 城市"
    ]
    hmm = HMM()
    hmm.train(train_data)  # 训练模型

    # 测试
    test_sentences = [
        "我是学霸",
        "你爱北京",
        "上海是首都",
        "你是中国老师"
    ]
    for sent in test_sentences:
        print(f"原句: {sent}")
        print("分词结果:", "/".join(hmm.cut(sent)))
        print("-" * 30)