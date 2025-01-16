# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 03:48:22 2024

@author: Lenovo
"""

import numpy as np

class sequenceTop:
    def __init__(self, simplicial_complex):
        self.simplicial_complex = simplicial_complex
        self.dimension = self.find_max_dimension()
        self.sorted_simplices = self.sort_and_group_simplices()
        self.simplices_count = self.count_simplices_per_dimension()
        self.boundary_matrices = self.compute_boundary_matrices()
        self.laplacian_matrices = self.compute_laplacian_matrices()
        self.min_positive_eigenvalues = self.compute_min_positive_eigenvalues()
        
    def find_max_dimension(self):
        # 找到最高维单形的维度
        max_dimension = max(len(simplex) - 1 for simplex in self.simplicial_complex)
        return max_dimension

    def sort_and_group_simplices(self):
        # 将不同维度的单形分组并按字典序排序
        grouped_simplices = [[] for _ in range(self.dimension + 1)]
        
        # 按维度分组
        for simplex in self.simplicial_complex:
            dimension = len(simplex) - 1
            grouped_simplices[dimension].append(tuple(simplex))
            
        # 对每一维的单形组进行排序
        for dim in range(len(grouped_simplices)):
            grouped_simplices[dim].sort()

        
        return grouped_simplices

    def count_simplices_per_dimension(self):
        # 统计每个维度的单形数量
        return [len(simplices) for simplices in self.sorted_simplices]

    def compute_boundary_matrices(self):
        # 计算每个维度的边界矩阵
        boundary_matrices = []
        
        for p in range(1, self.dimension + 1):
            # 当前维度的单形和低一维度的单形
            p_simplices = self.sorted_simplices[p]
            p_minus_1_simplices = self.sorted_simplices[p - 1]
            
            # 构造 (p-1) -> p 的边界矩阵
            matrix = np.zeros((len(p_minus_1_simplices), len(p_simplices)), dtype=int)
            
            for j, simplex in enumerate(p_simplices):
                # 构造当前 p 维单形的所有 (p-1) 维面
                for i, vertex in enumerate(simplex):
                    face = tuple(simplex[:i] + simplex[i + 1:])
                    if face in p_minus_1_simplices:
                        # 获取面在 (p-1) 维单形列表中的索引
                        row_index = p_minus_1_simplices.index(face)
                        # 设置矩阵值，使用 (-1)^i 以考虑方向
                        matrix[row_index, j] = matrix[row_index, j] + (-1) ** i
            
            boundary_matrices.append(matrix)
        
        return boundary_matrices

    def compute_laplacian_matrices(self):
        # 计算每个维度的拉普拉斯矩阵
        laplacian_matrices = []
        
        for p in range(self.dimension + 1):
            # 若存在 B_p 和 B_p^T
            if p > 0:
                B_p = self.boundary_matrices[p - 1]
                L_p_part1 = B_p.T @ B_p
            else:
                L_p_part1 = np.zeros((self.simplices_count[p], self.simplices_count[p]), dtype=int)
            
            # 若存在 B_{p+1} 和 B_{p+1}^T
            if p < self.dimension:
                B_p_plus_1 = self.boundary_matrices[p]
                L_p_part2 = B_p_plus_1 @ B_p_plus_1.T
            else:
                L_p_part2 = np.zeros((self.simplices_count[p], self.simplices_count[p]), dtype=int)
            
            # 计算 L_p
            L_p = L_p_part1 + L_p_part2
            laplacian_matrices.append(L_p)
        
        return laplacian_matrices
    
    def compute_min_positive_eigenvalues(self):
        # 计算每个维度的拉普拉斯矩阵的最小正特征值
        min_positive_eigenvalues = []
        
        for L in self.laplacian_matrices:
            # 计算特征值
            eigenvalues = np.linalg.eigvalsh(L)
            eigenvalues = [round(val, 6) for val in eigenvalues]
            
            # 提取所有正特征值，并取6位小数
            positive_eigenvalues = [val for val in eigenvalues if val > 0]
            
            # 找到最小的正特征值，如果没有正特征值，则返回0
            min_positive_eigenvalue = min(positive_eigenvalues, default=0)
            min_positive_eigenvalues.append(min_positive_eigenvalue)
        
        return min_positive_eigenvalues
    
    def compute_betti_numbers(self):
        """
        计算每个维度的 Betti 数。
    
        返回:
        list: 各维度的 Betti 数。
        """
        betti_numbers = []
    
        for n in range(self.dimension + 1):
            # 获取 n 维单形的个数
            num_n_simplices = self.simplices_count[n]
    
            # 计算 (n+1) 维边界矩阵的秩
            if n < self.dimension and len(self.boundary_matrices[n]) > 0:
                rank_b_n_plus_1 = np.linalg.matrix_rank(self.boundary_matrices[n])
            else:
                rank_b_n_plus_1 = 0
    
            # 计算 n 维边界矩阵的秩
            if n > 0 and len(self.boundary_matrices[n - 1]) > 0:
                rank_b_n = np.linalg.matrix_rank(self.boundary_matrices[n - 1])
            else:
                rank_b_n = 0
    
            # 计算 Betti 数
            betti_n = num_n_simplices - rank_b_n_plus_1 - rank_b_n
            betti_numbers.append(betti_n)
    
        return betti_numbers


    def get_result(self):
        # 返回按维度分组并排序后的单形列表、每个维度的单形数量、边界矩阵和拉普拉斯矩阵
        return {
            "grouped_simplices": self.sorted_simplices,
            "simplices_count": self.simplices_count,
            "boundary_matrices": self.boundary_matrices,
            "laplacian_matrices": self.laplacian_matrices,
            "min_positive_eigenvalues": self.min_positive_eigenvalues,
            "betti_numbers": self.compute_betti_numbers()
        }




# simplicial_complex = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [0, 0],[0,0,0], [1, 0,0],[0, 1, 2]]
# simplicial_complex = [[0], [1], [2],[0,1],[2,0],[2,1],[2,0,1]]
# st = sequenceTop(simplicial_complex)

# result = st.get_result()
# print("按维度分组的单形列表:")
# for dim, simplices in enumerate(result["grouped_simplices"]):
#     print(f"{dim} 维单形:", simplices)

# print("不同维度单形的个数:", result["simplices_count"])

# print("边界矩阵:")
# for dim, matrix in enumerate(result["boundary_matrices"], start=1):
#     print(f"{dim} 维边界矩阵:\n", matrix)

# print("拉普拉斯矩阵:")
# for dim, matrix in enumerate(result["laplacian_matrices"]):
#     print(f"{dim} 维拉普拉斯矩阵:\n", matrix)
    
# print("最小正特征值:")
# for dim, eigenvalue in enumerate(result["min_positive_eigenvalues"]):
#     print(f"{dim} 维拉普拉斯矩阵的最小正特征值:", eigenvalue)
    
# print("Betti 数:")
# for dim, betti in enumerate(result["betti_numbers"]):
#     print(f"{dim} 维 Betti 数:", betti)    
    
    
    
    
    
    
    
    
    
    
    
    
    