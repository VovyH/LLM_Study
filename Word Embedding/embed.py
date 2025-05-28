import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 定义水果和交通工具的列表
fruits = ["苹果", "香蕉", "橙子", "葡萄", "草莓"]
vehicles = ["汽车", "自行车", "飞机", "火车", "轮船"]

# 获取嵌入向量的函数
def get_embedding(text, url, headers, payload_template):
    payload = payload_template.copy()
    payload["input"] = text
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        print(f"Error: {response.status_code}")
        return None

# 配置API请求
url = "https://api.siliconflow.cn/v1/embeddings"
headers = {
    "Authorization": "Bearer <your_token>",
    "Content-Type": "application/json"
}
payload_template = {
    "model": "BAAI/bge-large-zh-v1.5",
    "encoding_format": "float"
}

# 获取所有嵌入向量
all_items = fruits + vehicles
embeddings = []
for item in all_items:
    embedding = get_embedding(item, url, headers, payload_template)
    if embedding is not None:
        embeddings.append(embedding)

# 将嵌入向量转换为numpy数组
embeddings = np.array(embeddings)

# 使用PCA或t-SNE进行降维
pca = PCA(n_components=2)
# tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = pca.fit_transform(embeddings)

# 绘制散点图
plt.figure(figsize=(10, 6))
for i, item in enumerate(all_items):
    if item in fruits:
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color='red', label='水果' if i == 0 else "")
    else:
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color='blue', label='交通工具' if i == len(fruits) else "")
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], item)

plt.legend()
plt.title("水果和交通工具的嵌入向量分类")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()