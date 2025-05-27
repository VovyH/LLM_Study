import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from collections import defaultdict
import numpy as np
from math import log

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class TokenizationVisualizer:
    def __init__(self, dictionary, corpus, sentence):
        self.dictionary = dictionary
        self.corpus = corpus
        self.sentence = sentence
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('N最短路径分词算法动态演示', fontsize=16, fontweight='bold')
        
        # 初始化各个子图
        self.ax_trie = self.axes[0, 0]  # 前缀树
        self.ax_tfidf = self.axes[0, 1]  # TF-IDF计算
        self.ax_graph = self.axes[1, 0]  # 分词图
        self.ax_paths = self.axes[1, 1]  # 路径结果
        
        # 计算数据
        self._build_trie()
        self._compute_tfidf()
        self._build_tokenization_graph()
        
    def _build_trie(self):
        """构建前缀树"""
        self.trie = {}
        self.trie_nodes = {}  # 用于可视化的节点信息
        
        for word in self.dictionary:
            node = self.trie
            path = ""
            for char in word:
                path += char
                if char not in node:
                    node[char] = {}
                    self.trie_nodes[path] = {'char': char, 'is_end': False, 'words': []}
                node = node[char]
                if path not in self.trie_nodes:
                    self.trie_nodes[path] = {'char': char, 'is_end': False, 'words': []}
            node['#'] = True
            self.trie_nodes[path]['is_end'] = True
            self.trie_nodes[path]['words'].append(word)
    
    def _compute_tfidf(self):
        """计算TF-IDF值"""
        self.tfidf = {}
        doc_count = len(self.corpus)
        word_doc_counts = defaultdict(int)
        
        # 计算文档频率
        for doc in self.corpus:
            seen_in_doc = set()
            for word in self.dictionary:
                if word in doc and word not in seen_in_doc:
                    word_doc_counts[word] += 1
                    seen_in_doc.add(word)
        
        # 计算TF-IDF
        for word in self.dictionary:
            tf_ratio = word_doc_counts[word] / doc_count if doc_count > 0 else 0
            idf = log((doc_count + 1) / (word_doc_counts[word] + 1)) + 1
            self.tfidf[word] = tf_ratio * idf
    
    def _build_tokenization_graph(self):
        """构建分词图"""
        self.graph = nx.DiGraph()
        self.possible_words = []
        
        # 搜索所有可能的词
        n = len(self.sentence)
        for i in range(n):
            node = self.trie
            for j in range(i, n):
                if self.sentence[j] not in node:
                    break
                node = node[self.sentence[j]]
                if '#' in node:
                    word = self.sentence[i:j+1]
                    weight = self.tfidf.get(word, 0)
                    self.possible_words.append((i, j+1, word, weight))
                    self.graph.add_edge(i, j+1, word=word, weight=weight)
        
        # 添加起始和结束节点
        self.graph.add_node('start')
        self.graph.add_node('end')
        for i, j, word, weight in self.possible_words:
            if i == 0:
                self.graph.add_edge('start', j, word=word, weight=weight)
            if j == len(self.sentence):
                self.graph.add_edge(i, 'end', word=word, weight=weight)
    
    def visualize_trie(self, step=0):
        """可视化前缀树"""
        self.ax_trie.clear()
        self.ax_trie.set_title('前缀树构建过程', fontweight='bold')
        
        # 创建前缀树图
        G = nx.DiGraph()
        pos = {}
        labels = {}
        colors = []
        
        # 添加根节点
        G.add_node('root')
        pos['root'] = (0, 0)
        labels['root'] = 'ROOT'
        colors.append('lightblue')
        
        # 添加节点和边
        level_nodes = defaultdict(list)
        for path, info in self.trie_nodes.items():
            G.add_node(path)
            level_nodes[len(path)].append(path)
            labels[path] = info['char']
            if info['is_end']:
                colors.append('lightgreen')
            else:
                colors.append('lightcoral')
            
            # 添加边
            if len(path) == 1:
                G.add_edge('root', path)
            else:
                parent = path[:-1]
                G.add_edge(parent, path)
        
        # 计算位置
        max_level = max(level_nodes.keys()) if level_nodes else 0
        for level, nodes in level_nodes.items():
            y = -level
            x_positions = np.linspace(-len(nodes)/2, len(nodes)/2, len(nodes))
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y)
        
        # 绘制图
        nx.draw(G, pos, ax=self.ax_trie, labels=labels, node_color=colors,
                node_size=800, font_size=8, font_weight='bold',
                arrows=True, arrowsize=20)
        
        # 添加词语标注
        word_text = "词典: " + ", ".join(self.dictionary)
        self.ax_trie.text(0.02, 0.98, word_text, transform=self.ax_trie.transAxes,
                         verticalalignment='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def visualize_tfidf(self, step=0):
        """可视化TF-IDF计算"""
        self.ax_tfidf.clear()
        self.ax_tfidf.set_title('TF-IDF权重计算', fontweight='bold')
        
        words = list(self.dictionary)
        weights = [self.tfidf[word] for word in words]
        
        # 创建柱状图
        bars = self.ax_tfidf.bar(range(len(words)), weights, 
                                color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        
        # 设置标签
        self.ax_tfidf.set_xticks(range(len(words)))
        self.ax_tfidf.set_xticklabels(words, rotation=45, ha='right')
        self.ax_tfidf.set_ylabel('TF-IDF权重')
        
        # 添加数值标签
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            self.ax_tfidf.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 添加计算公式说明
        formula_text = "TF-IDF = (文档频率/总文档数) × log((总文档数+1)/(词文档频率+1)) + 1"
        self.ax_tfidf.text(0.02, 0.98, formula_text, transform=self.ax_tfidf.transAxes,
                          verticalalignment='top', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def visualize_graph(self, step=0):
        """可视化分词图"""
        self.ax_graph.clear()
        self.ax_graph.set_title(f'分词路径图 - 句子: "{self.sentence}"', fontweight='bold')
        
        # 创建位置布局
        pos = {}
        sentence_len = len(self.sentence)
        
        # 为字符位置创建节点
        for i in range(sentence_len + 1):
            pos[i] = (i, 0)
        
        # 绘制字符
        for i, char in enumerate(self.sentence):
            self.ax_graph.text(i + 0.5, -0.3, char, ha='center', va='center',
                              fontsize=14, fontweight='bold',
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 绘制可能的词语路径
        for i, (start, end, word, weight) in enumerate(self.possible_words):
            y_offset = 0.3 + (i % 3) * 0.2  # 错开显示
            
            # 绘制弧线
            x_start, x_end = start, end
            x_mid = (x_start + x_end) / 2
            
            # 绘制路径
            self.ax_graph.annotate('', xy=(x_end, 0.1), xytext=(x_start, 0.1),
                                  arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                                connectionstyle=f"arc3,rad={y_offset}"))
            
            # 添加词语和权重标签
            self.ax_graph.text(x_mid, y_offset + 0.1, f'{word}\n({weight:.3f})',
                              ha='center', va='center', fontsize=9,
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # 设置坐标轴
        self.ax_graph.set_xlim(-0.5, sentence_len + 0.5)
        self.ax_graph.set_ylim(-0.5, 1.5)
        self.ax_graph.set_xticks(range(sentence_len + 1))
        self.ax_graph.set_xlabel('字符位置')
        self.ax_graph.grid(True, alpha=0.3)
    
    def visualize_paths(self, step=0):
        """可视化最佳路径"""
        self.ax_paths.clear()
        self.ax_paths.set_title('最佳分词路径', fontweight='bold')
        
        # 计算最佳路径（简化版本）
        paths = []
        # 这里可以实现完整的路径搜索算法
        # 为了演示，我们显示几个可能的路径
        
        if len(self.possible_words) >= 2:
            # 路径1: 商务/处女/干事
            path1 = [('商务', self.tfidf.get('商务', 0)), 
                    ('处女', self.tfidf.get('处女', 0)), 
                    ('干事', self.tfidf.get('干事', 0))]
            paths.append(path1)
            
            # 路径2: 商务处/女干事
            path2 = [('商务处', self.tfidf.get('商务处', 0)), 
                    ('女干事', self.tfidf.get('女干事', 0))]
            paths.append(path2)
        
        # 绘制路径
        y_positions = [0.8, 0.6, 0.4, 0.2]
        for i, path in enumerate(paths[:4]):
            if i < len(y_positions):
                y = y_positions[i]
                total_weight = sum(weight for _, weight in path)
                path_str = '/'.join([word for word, _ in path])
                
                # 绘制路径
                self.ax_paths.text(0.05, y, f'路径{i+1}: {path_str}',
                                  transform=self.ax_paths.transAxes,
                                  fontsize=12, fontweight='bold')
                self.ax_paths.text(0.05, y-0.05, f'总权重: {total_weight:.4f}',
                                  transform=self.ax_paths.transAxes,
                                  fontsize=10, color='blue')
        
        self.ax_paths.set_xlim(0, 1)
        self.ax_paths.set_ylim(0, 1)
        self.ax_paths.axis('off')
    
    def animate(self, frame):
        """动画函数"""
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        # 根据帧数显示不同阶段
        if frame < 30:  # 前缀树构建
            self.visualize_trie(frame)
            self.ax_tfidf.text(0.5, 0.5, '等待TF-IDF计算...', 
                              transform=self.ax_tfidf.transAxes, ha='center', va='center')
            self.ax_graph.text(0.5, 0.5, '等待分词图构建...', 
                              transform=self.ax_graph.transAxes, ha='center', va='center')
            self.ax_paths.text(0.5, 0.5, '等待路径搜索...', 
                              transform=self.ax_paths.transAxes, ha='center', va='center')
        elif frame < 60:  # TF-IDF计算
            self.visualize_trie()
            self.visualize_tfidf(frame - 30)
            self.ax_graph.text(0.5, 0.5, '等待分词图构建...', 
                              transform=self.ax_graph.transAxes, ha='center', va='center')
            self.ax_paths.text(0.5, 0.5, '等待路径搜索...', 
                              transform=self.ax_paths.transAxes, ha='center', va='center')
        elif frame < 90:  # 分词图构建
            self.visualize_trie()
            self.visualize_tfidf()
            self.visualize_graph(frame - 60)
            self.ax_paths.text(0.5, 0.5, '等待路径搜索...', 
                              transform=self.ax_paths.transAxes, ha='center', va='center')
        else:  # 路径搜索结果
            self.visualize_trie()
            self.visualize_tfidf()
            self.visualize_graph()
            self.visualize_paths(frame - 90)
        
        return []
    
    def create_animation(self, save_path=None):
        """创建动画"""
        anim = animation.FuncAnimation(self.fig, self.animate, frames=120, 
                                     interval=200, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
            print(f"动画已保存到: {save_path}")
        
        plt.tight_layout()
        plt.show()
        return anim

# 使用示例
if __name__ == "__main__":
    # 示例数据
    dictionary = {'商务处', '女干事', '商务', '处女', '干事'}
    corpus = [
        "商务处的职责是处理商务相关事务",
        "女干事在商务处工作", 
        "商务谈判需要处女干事参与",
        "干事们在商务处开会"
    ]
    sentence = "商务处女干事"
    
    # 创建可视化器
    visualizer = TokenizationVisualizer(dictionary, corpus, sentence)
    
    # 创建动画
    anim = visualizer.create_animation("tokenization_animation.gif")