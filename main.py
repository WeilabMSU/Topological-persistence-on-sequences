# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 23:28:59 2024

@author: Lenovo
"""

import pandas as pd
import numpy as np
import logging
from Bio import SeqIO
from collections import defaultdict
from delta_complex_geq import DeltaComplex
import time
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.cm import Paired


class KmerCounter:
    def __init__(self, n):
        """
        初始化 KmerCounter 类。

        参数:
        n (int): 子序列的最大长度。
        """
        self.n = n  # 存储最大子序列长度

    def count_kmers(self, sequence):
        """
        计算序列中所有长度小于等于n的子序列的频率。

        参数:
        sequence (str): 输入的DNA序列。

        返回:
        dict: 子序列及其对应的出现频率字典。
        """
        kmer_counts = defaultdict(int)  # 使用字典存储子序列的计数

        # 遍历所有可能的子序列长度（从 1 到 n）
        for length in range(1, self.n + 1):
            for i in range(len(sequence) - length + 1):
                kmer = sequence[i:i + length]
                kmer_counts[kmer] += 1

        return kmer_counts

    def kmer_f(self, kmer_counts, seq):
        """
        获取给定子序列的频率。

        参数:
        kmer_counts (dict): 已统计的子序列频率字典。
        seq (str): 输入的 k-mer 序列。

        返回:
        int: 该序列的出现频率。
        """
        return kmer_counts.get(seq, 0)  # 如果不存在，则返回 0


class TopologicalFeatureExtractor:
    def __init__(self, features, max_dim):
        """
        初始化 TopologicalFeatureExtractor 类。

        参数:
        features (dict): DeltaComplex 计算的拓扑特征字典。
        max_dim (int): 最大维度，提取的维度范围为 0 到 max_dim-1。
        """
        self.features = features
        self.max_dim = max_dim

    def reconstruct_festure(self, a, b):
        """
        重新构造特征，将 a 的值投影到自然数区间，并生成对应 b 的映射。

        参数:
        a (list): 键（如 keys）。
        b (list): 特征值（如 Betti 数或最小非零特征值）。

        返回:
        tuple: intervals 和对应的特征值向量。
        """
        a = np.sort(np.array(a))  # 确保 a 是递增的
        b = np.array(b)

        if len(a) != len(b):
            raise ValueError("Length of 'a' and 'b' must be the same")

        intervals = np.arange(0, int(np.ceil(a[-1])) + 1)
        v = np.zeros_like(intervals, dtype=b.dtype)

        for i, n in enumerate(intervals):
            idx = np.searchsorted(a, n, side='left')
            idx = np.clip(idx, 0, len(b) - 1)
            v[i] = b[idx]

        return v

    def extract_topological_features(self):
        keys = list(self.features.keys())
        betti_vectors = []
        eigen_vectors = []

        for k in range(self.max_dim):
            betti_k_dim = [
                self.features[key]['betti_numbers'][k] if len(self.features[key]['betti_numbers']) > k else 0
                for key in keys
            ]
            eigen_k_dim = [
                self.features[key]['min_positive_eigenvalues'][k] if 
                len(self.features[key]['min_positive_eigenvalues']) > k else 0
                for key in keys
            ]

            # 使用 reconstruct_festure 重新映射
            betti_vector = self.reconstruct_festure(keys, betti_k_dim)
            eigen_vector = self.reconstruct_festure(keys, eigen_k_dim)

            betti_vectors.append(betti_vector)
            eigen_vectors.append(eigen_vector)

        return betti_vectors, eigen_vectors


def upgma_clustering(feature_matrix, labels, colors, metric='cityblock', title='UPGMA Dendrogram', output_file=None):
    """
    Perform UPGMA clustering on a feature matrix and visualize the dendrogram with specified colors for labels.

    Parameters:
    - feature_matrix (numpy.ndarray): A 2D array where each row is a feature vector.
    - labels (list of str): Labels for each row in the feature matrix.
    - colors (dict): Mapping of labels to their respective colors.
    - metric (str): Distance metric to use (default: 'euclidean').
    - title (str): Title for the dendrogram plot.
    - output_file (str): File path to save the dendrogram image (optional).

    Returns:
    - linkage_matrix (numpy.ndarray): Linkage matrix generated by hierarchical clustering.
    """
    # Compute pairwise distances
    condensed_dist = pdist(feature_matrix, metric=metric)
    linkage_matrix = sch.linkage(condensed_dist, method='average')

    # Create the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram = sch.dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=270,
        leaf_font_size=9,
        link_color_func=lambda k: 'black'  # Default black for branches
    )

    # Assign colors to leaves (labels)
    ax = plt.gca()
    x_labels = ax.get_xmajorticklabels()
    
    for i, label in enumerate(x_labels):
        text = label.get_text()
        label.set_color('black')  # Set text color to black
        
        # Get background color for each label based on gene's category
        color = colors.get(text, 'gray')  # Default to gray if not found in colors mapping
        label.set_bbox(dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.5'))

    # Remove y-axis labels (if desired)
    ax.get_yaxis().set_visible(False)
    
    # Finalize the plot
    if title:
        plt.title(title)
    plt.tight_layout()

    # Save the plot to a file if specified
    if output_file:
        plt.savefig(output_file, format=output_file.split('.')[-1], dpi=300)
        print(f"Dendrogram saved as {output_file}")

    plt.show()
    return linkage_matrix



# 设置 CSV 文件路径
input_file = "ebolavirus_record.csv"

# 读取 CSV 文件，从第二行开始，提取第2列和第4列
df = pd.read_csv(input_file, header=0, usecols=[1, 4])

# 构建标签字典
label_gene_origin = {}
for index, row in df.iterrows():
    gene_id = row[0]  # 第2列数据（基因ID）
    label = row[1]    # 第4列数据（标签）
    label_gene_origin[gene_id] = label
    

# 设置 NumPy 输出选项，取消长度限制
np.set_printoptions(threshold=np.inf)

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 输入和输出文件设置
input_file = "ebolavirus_sequences.fasta"
output_file = "topological_features.txt"

# 初始化参数
max_dim = 4  # 最大子序列长度

# 用于存储所有特征的列表
features = []


# 逐条读取 DNA 序列并处理
start_time = time.time()
sequence_count = 0

for record in SeqIO.parse(input_file, "fasta"):
    sequence_count += 1
    sequence = str(record.seq)
    logging.info(f"正在处理序列 {sequence_count}，长度为 {len(sequence)}")

    # 创建 KmerCounter 对象并统计 k-mer
    kmer_counter = KmerCounter(max_dim)
    kmer_counts = kmer_counter.count_kmers(sequence)

    # 创建 DeltaComplex 对象
    complex_obj = DeltaComplex(lambda seq: kmer_counter.kmer_f(kmer_counts, seq), max_dim)
    topological_features = complex_obj.compute_topological_features()

    # 创建 TopologicalFeatureExtractor 对象
    extractor = TopologicalFeatureExtractor(topological_features, max_dim=max_dim)
    betti_vectors, eigen_vectors = extractor.extract_topological_features()

    # 存储特征到 features 列表
    features.append({
        "sequence_id": record.id,
        "betti_vectors": betti_vectors,
        "eigen_vectors": eigen_vectors
    })

    logging.info(f"序列 {sequence_count} 特征已处理完成")

end_time = time.time()
logging.info(f"所有序列处理完毕，总序列数: {sequence_count}，总耗时: {end_time - start_time:.2f} 秒")

# 提取 Eigen Vectors 并补齐长度
used_feature_vectors = [feature['eigen_vectors'][2] for feature in features if len(feature['eigen_vectors']) > 2]
max_length = max(len(vec) for vec in used_feature_vectors)
padded_vectors = [np.pad(vec, (0, max_length - len(vec)), 'constant') for vec in used_feature_vectors]
feature_matrix = np.array(padded_vectors)
print(feature_matrix.shape)
# 创建标签数组
label_gene_origin = {
    'FJ217161': 'BDBV',
    'KC545393': 'BDBV',
    'KC545395': 'BDBV',
    'KC545394': 'BDBV',
    'KC545396': 'BDBV',
    'FJ217162': 'TAFV',
    'AF522874': 'RESTV',
    'AB050936': 'RESTV',
    'JX477166': 'RESTV',
    'FJ621585': 'RESTV',
    'FJ621583': 'RESTV',
    'JX477165': 'RESTV',
    'FJ968794': 'SUDV',
    'KC242783': 'SUDV',
    'EU338380': 'SUDV',
    'AY729654': 'SUDV',
    'JN638998': 'SUDV',
    'KC545389': 'SUDV',
    'KC545390': 'SUDV',
    'KC545391': 'SUDV',
    'KC545392': 'SUDV',
    'KC589025': 'SUDV',
    'KC242801': 'EBOV',
    'NC_002549': 'EBOV',
    'KC242791': 'EBOV',
    'KC242792': 'EBOV',
    'KC242793': 'EBOV',
    'KC242794': 'EBOV',
    'AY354458': 'EBOV',
    'KC242796': 'EBOV',
    'KC242799': 'EBOV',
    'KC242784': 'EBOV',
    'KC242786': 'EBOV',
    'KC242787': 'EBOV',
    'KC242789': 'EBOV',
    'KC242785': 'EBOV',
    'KC242790': 'EBOV',
    'KC242788': 'EBOV',
    'KC242800': 'EBOV',
    'KM034555': 'EBOV',
    'KM034562': 'EBOV',
    'KM233039': 'EBOV',
    'KM034557': 'EBOV',
    'KM034560': 'EBOV',
    'KM233050': 'EBOV',
    'KM233053': 'EBOV',
    'KM233057': 'EBOV',
    'KM233063': 'EBOV',
    'KM233072': 'EBOV',
    'KM233110': 'EBOV',
    'KM233070': 'EBOV',
    'KM233099': 'EBOV',
    'KM233097': 'EBOV',
    'KM233109': 'EBOV',
    'KM233096': 'EBOV',
    'KM233103': 'EBOV',
    'KJ660346': 'EBOV',
    'KJ660347': 'EBOV',
    'KJ660348': 'EBOV',
}



color_palette = [to_hex(Paired(0)), to_hex(Paired(2)),to_hex(Paired(4)),to_hex(Paired(6)),to_hex(Paired(8))] 

# 将每个标签类别分配给调色板中的颜色
unique_labels = sorted(set(label_gene_origin.values()))
if len(unique_labels) > len(color_palette):
    raise ValueError("The number of unique labels exceeds the available colors in the palette.")

label_to_color = {label: color_palette[i] for i, label in enumerate(unique_labels)}

# 将基因序列对应到颜色
gene_to_color = {gene: label_to_color[label_gene_origin[gene]] for gene in label_gene_origin.keys()}

# 基因序列名称和颜色
sorted_genes = list(label_gene_origin.keys())

# 调用函数进行聚类可视化
upgma_clustering(
    feature_matrix=feature_matrix,
    labels=sorted_genes,
    colors=gene_to_color,
    metric='cityblock',
    title=None,
    output_file='ebola_clustering_colored.png'
)


