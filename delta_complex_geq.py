# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 06:20:13 2024

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 03:13:05 2024

@author: Lenovo
"""

from itertools import product
from stopology import sequenceTop
class DeltaComplex:
    def __init__(self, f, n, X=None):
        """
        初始化DeltaComplex类
        
        参数：
        X (list): 有限集合X。
        f (function): 输入为X中元素组成的长度小于等于n的sequence的函数。
        n (int): 序列的最大长度。
        """
        self.X = sorted(X) if X is not None else ['A', 'C', 'G', 'T']
        self.n = n
        
        # 使用 delta_closure 将 f 转换为 make_tilde_f
        self.make_tilde_f = self.delta_closure(f)
        
        # 生成复形
        self.complex = self.generate_complex()

    def delta_closure(self, f):
        """
        对原始函数 f 进行闭包操作，生成修改后的函数 tilde_f。
        
        输出:
        function: 经过闭包操作后的新函数（tilde_f）。
        """
        def get_faces(seq):
            """
            获取某个序列的所有面（删除一个元素后的所有子序列）。
            """
            faces = []
            for i in range(len(seq)):
                # 删除第i个元素后的子序列
                face = seq[:i] + seq[i+1:]
                faces.append(face)
            return faces

        def tilde_f(seq):
            """
            计算修改后的 f，返回 \widetilde{f}(\sigma)
            """
            # 如果序列长度为 1，直接返回 f(seq)
            if len(seq) == 1:
                return f(seq)
            
            # 获取所有面
            faces = get_faces(seq)
            
            # 递归计算每个面上的函数值
            sup_f = min(self.make_tilde_f(face) for face in faces)
            
            # 返回 \max(\sup_{\tau} f(\tau), f(\sigma))
            return min(sup_f, f(seq))
        
        return tilde_f  # 返回修改后的函数tilde_f

    def generate_sequences(self):
        """
        生成集合 X 中元素的所有长度小于等于 n 的序列，作为字符串形式
        """
        sequences = []
        for length in range(1, self.n + 1):
            sequences.extend(''.join(seq) for seq in product(self.X, repeat=length))
        return sequences

    def generate_complex(self):
        """
        生成 Delta 复形
        
        输出:
        dict: key 为函数值 a，value 为包含所有函数值小于（或大于）等于 a 的 sequence 组成的复形数组。
        """
        sequences = self.generate_sequences()
        
        # 使用 make_tilde_f 对序列进行运算
        function_values = {seq: self.make_tilde_f(seq) for seq in sequences}
        
        # 找到所有函数值的唯一值，从小到大排序
        unique_values = sorted(set(function_values.values()), reverse=False)
        
        # 构建复形数组，对于每一个函数值 a，将函数值大于等于 a 的 sequence 提取出来
        delta_complex = {}
        for a in unique_values:
            complex_a = [seq for seq, val in function_values.items() if val <= a]
            delta_complex[a] = [self.sequence_to_indices(seq) for seq in complex_a]
        
        return delta_complex
    
    def sequence_to_indices(self, sequence):
        """
        将字符串序列转换为 X 中元素编号组成的数组表示
        
        参数:
        sequence (str): X 中元素组成的字符串序列。
        
        输出:
        list: 对应序列的编号组成的数组。
        """
        return [self.X.index(elem) for elem in sequence]
    
    def compute_topological_features(self):
        """
        计算 Delta 复形的拓扑特征，包括 Betti 数和最小正特征值，返回一个字典，其中键是函数值，值是包含 Betti 数和最小正特征值的列表。
        
        输出:
        dict: 键为函数值，值为包含 Betti 数和最小正特征值的字典。
        """
        topological_features = {}
        
        for a, complex_a in self.complex.items():
            # 使用 sequenceTop 计算 Betti 数和最小正特征值
            sequence_top = sequenceTop(complex_a)
            betti_numbers = sequence_top.compute_betti_numbers()
            min_positive_eigenvalues = sequence_top.compute_min_positive_eigenvalues()
            
            # 将 Betti 数和最小正特征值存入字典
            topological_features[a] = {
                "betti_numbers": betti_numbers,
                "min_positive_eigenvalues": min_positive_eigenvalues
            }
        
        return topological_features


# # 测试
# n = 3

# def f(seq):
#     num = {
#         'A': 5, 'C': 5, 'G': 6, 'T': 6,
#         'AA': 2, 'AC': 1, 'AG': 1, 'AT': 1,
#         'CA': 0, 'CC': 1, 'CG': 0, 'CT': 4,
#         'GA': 2, 'GC': 1, 'GG': 3, 'GT': 0,
#         'TA': 1, 'TC': 2, 'TG': 1, 'TT': 1,
#         'AAA': 0, 'AAC': 1, 'AAG': 0, 'AAT': 1,
#         'ACA': 0, 'ACC': 0, 'ACG': 0, 'ACT': 1,
#         'AGA': 1, 'AGC': 0, 'AGG': 0, 'AGT': 0,
#         'ATA': 0, 'ATC': 0, 'ATG': 0, 'ATT': 0,
#         'CAA': 0, 'CAC': 0, 'CAG': 0, 'CAT': 0,
#         'CCA': 0, 'CCC': 0, 'CCG': 0, 'CCT': 1,
#         'CGA': 0, 'CGC': 0, 'CGG': 0, 'CGT': 0,
#         'CTA': 1, 'CTC': 1, 'CTG': 1, 'CTT': 1,
#         'GAA': 2, 'GAC': 0, 'GAG': 0, 'GAT': 0,
#         'GCA': 0, 'GCC': 0, 'GCG': 0, 'GCT': 1,
#         'GGA': 1, 'GGC': 0, 'GGG': 2, 'GGT': 0,
#         'GTA': 1, 'GTC': 0, 'GTG': 0, 'GTT': 0,
#         'TAA': 0, 'TAC': 0, 'TAG': 1, 'TAT': 0,
#         'TCA': 0, 'TCC': 1, 'TCG': 0, 'TCT': 1,
#         'TGA': 0, 'TGC': 1, 'TGG': 0, 'TGT': 0,
#         'TTA': 0, 'TTC': 1, 'TTG': 0, 'TTT': 0
#     }

#     return num.get(seq, 0)


# # def f(seq):
# #     return len(seq)

# # 初始化DeltaComplex类
# complex_obj = DeltaComplex(f, n)

# # 输出复形
# print("complex::",complex_obj.complex[6])