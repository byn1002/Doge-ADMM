import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse
from time import process_time
import cvxpy as cp
import copy
from PIL import Image

def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
# Generate Delta

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

    

import networkx as nx

from scipy.spatial import ConvexHull
def plot_single_partition(ax, rows, indices, partition_number, colors, m, n):
    """
    plot partition
    """
    for idx, row_index in enumerate(indices):
        row = rows[row_index]
        nodes = []
        for col in row:
            x = col % n 
            y = col // n 
            ax.scatter(x, y, color=colors[idx], s=200, label=f'Row {row_index}' if col == row[0] else "")
            nodes.append((x, y))
        
        if len(nodes) > 2:
            nodes = np.array(nodes)
            hull = ConvexHull(nodes)

            for simplex in hull.simplices:
                ax.plot(nodes[simplex, 0], nodes[simplex, 1], color=colors[idx])
            ax.fill(nodes[hull.vertices,0], nodes[hull.vertices,1], color=colors[idx], alpha=0.3)
        if len(nodes) == 2:
            ax.plot([nodes[0][0], nodes[1][0]], [nodes[0][1], nodes[1][1]], color=colors[idx])

    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.5, m-0.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(m))
    ax.grid(True)

    ax.set_title(f"Partition {partition_number}",fontsize=30)




def generate_classes_2(m, n, mods):
    """
    generate classes for case B=3
    """
    # Step 1: Generate the grid dimensions
    grid_row = 2 * m 
    grid_col = 2 * n 

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


def generate_classes_1(m, n, mods):
    """
    generate classes for case B=2
    """
    # Step 1: Generate the grid dimensions
    grid_row = m
    grid_col = n

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

    classification = [[] for _ in range(5)]  # Initialize a list for each class
    rows = [[] for _ in range(5)]
    for i in range(grid_row):
        for j in range(grid_col):
            mod_pair = ((2*i+j)%5, (i-2*j)%5)
            if mod_pair in mods:
                class_label = mods.index(mod_pair) + 1
                temp_k = i
                temp_l = j
                
                if 0 <= temp_k < m and 0 <= temp_l < n:# and temp_k + 1 < m:
                    classification[class_label - 1].append((temp_k, temp_l))
                    node_index = temp_l * m + temp_k
                    rows[class_label - 1].append(node_index)
    return rows


def generate_classes_0(m, n, mods):
    """
    generate classes for case B=1
    """
    # Step 1: Generate the grid dimensions
    grid_row = 2 * m 
    grid_col = 2 * n 

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

    classification = [[] for _ in range(4)]  # Initialize a list for each class
    
    for i in range(grid_row):
        for j in range(grid_col):
            mod_pair = ((i+j)%4, (i-j)%4)
            if mod_pair in mods:
                class_label = mods.index(mod_pair) + 1
                temp_k = (i - 1) // 2
                temp_l = j // 2
                
                if 0 <= temp_k < m and 0 <= temp_l < n :# and temp_k + 1 < m
                    classification[class_label - 1].append((temp_k, temp_l))
    class_edges = []
    for class_idx, class_points in enumerate(classification[:2]):
        edges = []
        for (x, y) in class_points:
            if G.has_edge((x, y), (x + 1, y)):
                edge_number = int(G[(x, y)][(x + 1, y)]['label'])-1
                edges.append(edge_number)
        class_edges.append(edges) 
    for class_idx, class_points in enumerate(classification[2:4], start=8):
        edges = []
        for (x, y) in class_points:
            if G.has_edge((x, y), (x, y + 1)):
                edge_number = int(G[(x, y)][(x, y + 1)]['label'])-1
                edges.append(edge_number)
        class_edges.append(edges)    
    return class_edges, classification






# Some algorithms for trend filtering on graphs

# The ADMM_on_graphs function may not always run correctly when k is even due to the convex optimization step for updating z using CVXPY. 
def ADMM_on_graphs(image,k,lambd,rho,max_iter=100,tol=1e-4):
    """
    Runs the ADMM (Alternating Direction Method of Multipliers) algorithm on graph structures derived from the image 
    for trend filtering on the graph.

    Parameters:
    - image: Input image data as a 2D array
    - k: Degree of the filter, determining the level of smoothing (even or odd values affect the calculation method)
    - lambd: Regularization parameter
    - rho: Penalty parameter in ADMM
    
    Returns:
    - beta: The filtered result after applying ADMM
    - i: Number of iterations performed
    - primal_residuals: List of primal residuals across iterations
    - dual_residuals: List of dual residuals across iterations
    """
    begin = process_time()
    y = image.flatten(order='F')
    n = y.shape[0]

    L = generate_L(image)
    beta = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    primal_residuals = []
    dual_residuals = []
    if k%2 == 0:
        S=generate_D(image)
        q = k//2
    else:
        S=L
        q = (k+1)//2
    Lq=np.linalg.matrix_power(L, q)
    L2q=np.linalg.matrix_power(L, 2*q)

    for i in range(max_iter):
        # Update beta
        b = y + rho * Lq.T @ (z + u)
        A= np.eye(n) + rho * L2q
        beta = np.linalg.solve(A, b)

        # Update z
        if k % 2 == 0:
            z_var = cp.Variable(n)
            objective = cp.Minimize((lambd / rho) * cp.norm1(S @ z_var) + 0.5 * cp.sum_squares(Lq @ beta - z_var - u))
            problem = cp.Problem(objective)
            result = problem.solve(solver=cp.ECOS)

            if z_var.value is not None:
                z = z_var.value
            else:
                print("Warning: Solver failed to find a feasible solution for z. Using previous z value.")
                z = z
        else:
            z = soft_thresholding(Lq @ beta - u, lambd / rho)
            
        # Update dual variable u
        u = u - Lq @ beta + z

        # Calculate primal and dual residuals
        primal_residual = Lq @ beta - z
        dual_residual = rho * Lq @ (z - u)
        primal_residuals.append(np.linalg.norm(primal_residual))
        dual_residuals.append(np.linalg.norm(dual_residual))

        # Check for convergence
        if primal_residuals[-1] < tol:
            break
    end = process_time()
    print("----- ADMM on Graphs-----")
    print("Time execution:", end - begin)    
    return beta, i, primal_residuals, dual_residuals

# Fast Newton
from scipy.sparse.linalg import cg 
def projected_newton_method(y, Delta, lambd, max_iter=100, tol=1e-4):
    """
    Implements the projected Newton method for solving the dual problem.
    
    Parameters:
    - y: numpy array, target vector.
    - Delta: numpy array, the matrix Δ^(k+1).
    - lambd: float, constraint value for the l1 norm.
    - max_iter: int, maximum number of iterations.
    - tol: float, tolerance for convergence.
    
    Returns:
    - v: solution vector.
    - beta: solution of the original problem.
    """
    # Initialize variables
    v = np.zeros(Delta.shape[1])  # Initialize v
    I = np.arange(Delta.shape[1])  # Indices of all elements

    for iteration in range(max_iter):
        # Compute gradient g and reduced Hessian H
        g = Delta.T @ (Delta @ v - y)  # Gradient of the dual objective
        Hv = Delta.T @ Delta  # Reduced Hessian, typically sparse

        # Solve the linear system using a conjugate gradient solver
        # Here we use cg (conjugate gradient) with tolerance for improved conditioning
        a, _ = cg(Hv, g, tol=tol)

        # Perform update for v: v = v - a
        v -= a

        # Project v to satisfy the l1 constraint ||v||_1 <= lambd
        # Using soft thresholding as projection
        v = project_l1(v, lambd)

        # Check convergence
        if np.linalg.norm(g, 2) < tol:
            print(f"Converged in {iteration} iterations.")
            break

    # Compute beta from v
    beta = y - Delta.T @ v
    return v, beta

def project_l1(v, lambd):
    """Projects vector v onto the l1-norm ball of radius lambd."""
    if np.sum(np.abs(v)) <= lambd:
        return v
    else:
        u = np.abs(v)
        if u.sum() == 0:
            return v
        w = np.sort(u)[::-1]
        sv = np.cumsum(w)
        rho = np.where(w > (sv - lambd) / (np.arange(1, len(w) + 1)))[0][-1]
        theta = (sv[rho] - lambd) / (rho + 1)
        return np.sign(v) * np.maximum(u - theta, 0)
    
# TVmodel
def solve_case_1(w,l,gamma):
    pho = l/gamma
    G = np.array([[-1,1,0],[0,-1,1]])
    B = np.linalg.inv(np.dot(G,G.T))
    B = np.dot(B,np.dot(G,w.T))
    pho_max = np.linalg.norm(B)
    if pho > pho_max:
        B = np.linalg.inv(np.dot(G,G.T))
        B = np.dot(G.T,B)
        B = np.dot(B,G)
        u = np.dot(np.identity(3)-B,w.T)

    else:
        U, S, VT = np.linalg.svd(G)
        S_full = np.zeros((2, 3))
        S_full[:, :len(S)] = np.diag(S)
        w_hat = np.dot(S_full,np.dot(VT,w.T))
        w_hat = w_hat.T
        c0 = (pho**8)*9-(pho**6)*(w_hat[0]**2+(w_hat[1]**2)*9)
        c1 = 2*(pho**6)*(3+9) - 2*(pho**4)*(w_hat[0]**2+w_hat[1]**2*3)
        c2 = pho**4*(9+1+12)-pho**2*(w_hat[0]**2+w_hat[1]**2)
        c3 = 2*pho**2*4

        coefficients = [1,c3,c2,c1,c0]
        roots = np.roots(coefficients)
        positive_real_roots = [root.real for root in roots if np.isreal(root) and root.real > 0]
        
        alpha =np.min(positive_real_roots)
        B = np.linalg.inv(np.dot(G,G.T)+alpha/(pho**2)*np.identity(2))
        B = np.dot(G.T,B)
        B = np.dot(B,G)
        u = np.dot(np.identity(3)-B,w.T)

    return u.T

def solve_case_2(w,l,gamma):
    pho = l/gamma
    if w[0]>w[1] + 2*pho:
        u = np.array([w[0]-pho,w[1]+pho])
    elif w[0] < w[1]-2*pho:
        u = np.array([w[0]+pho,w[1]-pho])
    else:
        u = np.array([(w[0]+w[1])/2,(w[0]+w[1])/2])
    return u.T

def minfunc(V,l,gamma,k):
    m,n = V.shape
    X = copy.deepcopy(V)
    for p in range(m):
        for q in range(n):
            if (p-q) % 3 != k:
                continue
            if (p < m-1) and (q < n-1):
                w = np.array([V[p,q+1],V[p,q],V[p+1,q]])
                u = solve_case_1(w,l,gamma)
                X[p,q+1] = u[0]
                X[p,q] = u[1]
                X[p+1,q] = u[2]
            
            if (p == m-1) and (q < n-1):
                w = np.array([V[p,q+1],V[p,q]])
                u = solve_case_2(w,l,gamma)
                X[p,q+1] = u[0]
                X[p,q] = u[1]
            
            if (p < m-1) and (q == n-1):
                w = np.array([V[p,q],V[p+1,q]])
                u = solve_case_2(w,l,gamma)
                X[p,q] = u[0]
                X[p+1,q] = u[1]
    return X

def FAD(Y,l,gamma,max_iter=100):
    m,n = Y.shape
    Z = copy.deepcopy(Y)
    X = [copy.deepcopy(Y), copy.deepcopy(Y), copy.deepcopy(Y)]
    theta = [np.zeros((m,n)),np.zeros((m,n)),np.zeros((m,n))]
    V = [np.zeros((m,n)),np.zeros((m,n)),np.zeros((m,n))]
    for iter in range(max_iter):

        for k in range(3):
            V[k] = Z-theta[k]
            X[k] = minfunc(V[k],l,gamma,k)
        Z = (Y+theta[0]+theta[1]+theta[2]+X[0]+X[1]+X[2])/(1+3*gamma)

        for k in range(3):
            theta[k] = theta[k] + X[k] - Z
    
    return Z


