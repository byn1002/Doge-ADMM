import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse

def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def generate_L(image): # index=i*n+j
    m, n = image.shape
    num_pixels = m * n

    A = lil_matrix((num_pixels, num_pixels)) # adjacency matrix
    D = np.zeros((num_pixels, num_pixels)) # degree matrix

    def pixel_index(i, j):
        return i * n + j

    # build A
    for i in range(m):
        for j in range(n):
            index = pixel_index(i, j)

            # 4-neighborhood
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    neighbor_index = pixel_index(ni, nj)
                    A[index, neighbor_index] = 1
                    A[neighbor_index, index] = 1

    # calculate D
    for i in range(num_pixels):
        D[i, i] = A[i].sum()

    # calculate L
    L = D - A.toarray()
    return L

def generate_D(image): # index=i*n+j
    m, n = image.shape
    num_nodes = m * n
    edges = []

    def pixel_index(i, j):
        return i * n + j

    # build edges
    for i in range(m):
        for j in range(n):
            current_index = pixel_index(i, j)

            # right edge
            if j + 1 < n:
                right_index = pixel_index(i, j + 1)
                edges.append((current_index, right_index))

            # down edge
            if i + 1 < m:
                down_index = pixel_index(i + 1, j)
                edges.append((current_index, down_index))

    num_edges = len(edges)
    
    D = lil_matrix((num_edges, num_nodes))

    # generate D
    for edge_index, (start_node, end_node) in enumerate(edges):
        D[edge_index, start_node] = -1
        D[edge_index, end_node] = 1

    return D.toarray()

def generate_delta(D, L, k):
    delta = []
    if (k+1) % 2 == 0:
        delta=D@np.linalg.matrix_power(L, (k-1)//2)
    else:
        delta=np.linalg.matrix_power(L, (k)//2)
    return delta


# 回溯算法
def can_add_to_partition(partition_set, row_set):
    """检查是否可以将行添加到当前分区，而不引入重复元素"""
    return partition_set.isdisjoint(row_set)

def backtrack(rows, partitions, partition_sets, index, row_indices):
    """回溯算法，尝试将 rows 分为 max_part 个不重复的分区"""
    if index == len(rows):
        return True  # 所有行都成功分配
    current_row = rows[index]
    current_row_set = set(current_row)
    
    for i in range(len(partitions)):
        if can_add_to_partition(partition_sets[i], current_row_set):
            partitions[i].extend(current_row)
            partition_sets[i].update(current_row_set)
            row_indices[i].append(index)  # 记录分区中的行索引
            
            if backtrack(rows, partitions, partition_sets, index + 1, row_indices):
                return True
            
            # 回溯
            partitions[i] = partitions[i][:-len(current_row)]
            partition_sets[i].difference_update(current_row_set)
            row_indices[i].pop()

    return False  # 无法找到有效分配

def divide_rows_into_partitions(rows, max_part):
    partitions = [[] for _ in range(max_part)]
    partition_sets = [set() for _ in range(max_part)]  # 使用集合存储分区中的元素
    row_indices = [[] for _ in range(max_part)]  # 初始化行索引
    if backtrack(rows, partitions, partition_sets, 0, row_indices):
        return partitions, row_indices
    else:
        return None, None  # 无法找到有效分配

def get_partition_columns(partitions):
    """计算每个分区中所有元素的唯一列集合，并存储为列表"""
    partition_columns = []
    for partition in partitions:
        columns = sorted(set(partition))  # 使用 sorted 保持输出整齐
        partition_columns.append(columns)
    return partition_columns



def generate_grid_and_assign_classes(m, n):
    # Step 1: Generate the grid
    grid_row = 2 * m - 1
    grid_col = 2 * n - 1
    # Step 2: Define the eight classes
    groups = [np.zeros((m, n), dtype=int) for _ in range(8)]
    mods = [
        (1,1),(3,3),(5,5),(7,7),
        (3,7),(5,1),(7,3),(1,5)
    ]
    for i in range(grid_row):
        for j in range(grid_col):
            mod_pair = ((i + j) % 8, (i - j) % 8)
            if mod_pair in mods:
                class_label = mods.index(mod_pair) + 1
                class_temp = np.zeros((m, n), dtype=int)
                temp_k=(i-1)//2
                temp_l=j//2
                class_temp[temp_k, temp_l] = class_label
                if temp_k+1<m:
                    class_temp[temp_k+1, temp_l] = class_label
                groups[class_label-1] += class_temp
    return groups        


import networkx as nx
def generate_classes(m, n, mods):
    # Step 1: Generate the grid dimensions
    grid_row = 2 * m - 1
    grid_col = 2 * n - 1

    # Step 2: Generate the graph
    G = nx.Graph()
    for y in range(n):
        for x in range(m):
            G.add_node((x, y))
    # add edges and labels
    edge_id = 1
    for y in range(n):
        for x in range(m):
            if x < m - 1:  
                G.add_edge((x, y), (x + 1, y), label=f'{edge_id}')
                edge_id += 1
            if y < n - 1:  
                G.add_edge((x, y), (x, y + 1), label=f'{edge_id}')
                edge_id += 1

    classification = [[] for _ in range(16)]  # Initialize a list for each class
    
    for i in range(grid_row):
        for j in range(grid_col):
            mod_pair = ((i + j) % 8, (i - j) % 8)
            if mod_pair in mods:
                class_label = mods.index(mod_pair) + 1
                temp_k = (i - 1) // 2
                temp_l = j // 2
                
                if 0 <= temp_k < m and 0 <= temp_l < n and temp_k + 1 < m:
                    classification[class_label - 1].append((temp_k, temp_l))

    for i in range(grid_row):
        for j in range(grid_col):
            mod_pair = ((i + j) % 8, (i - j) % 8)
            if mod_pair in mods:
                class_label = mods.index(mod_pair) + 1
                temp_k = (i - 1) // 2
                temp_l = j // 2
                
                if 0 <= temp_k < m and 0 <= temp_l < n and temp_k + 1 < m:
                    classification[class_label +7].append((temp_l,temp_k ))
    class_edges = []
    for class_idx, class_points in enumerate(classification[:8]):
        edges = []
        for (x, y) in class_points:
            if (x + 1, y) in G:
                if G.has_edge((x, y), (x + 1, y)):
                    edge_number = int(G[(x, y)][(x + 1, y)]['label'])-1
                    edges.append(edge_number)
        class_edges.append(edges) 
    for class_idx, class_points in enumerate(classification[8:16], start=8):
        edges = []
        for (x, y) in class_points:
            if (x , y+1) in G:
                if G.has_edge((x, y), (x, y + 1)):
                    edge_number = int(G[(x, y)][(x, y + 1)]['label'])-1
                    edges.append(edge_number)
        class_edges.append(edges)    
    return class_edges


# print the partition results 
def plot_single_partition(ax, rows, indices, partition_number, colors,m,n):
    for idx, row_index in enumerate(indices):
        row = rows[row_index]
        for col in row:
            x = col % n  # col
            y = col // n  # row
            ax.scatter(x, y, color=colors(idx),s=200, label=f'Row {row_index}' if col == row[0] else "")

    ax.set_xlim(-0.5, n+0.5)
    ax.set_ylim(-0.5, m)
    ax.invert_yaxis()  
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    ax.grid(True)

    # merge legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
    ax.set_title(f"Partition {partition_number}")