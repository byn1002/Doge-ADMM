import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse
import matplotlib.pyplot as plt
import networkx as nx
from tfgraph import *
import copy
import time
import cvxpy as cp

def loss(y, beta, delta_loss, l):
    return 0.5 * np.linalg.norm(y - beta) ** 2 + l * np.linalg.norm(delta_loss @ beta, ord=1)

def step1(V, l, rho, k, shape): # 并行版本
    """
    通过向量化并行运算更新矩阵 W。

    参数：
        V: numpy 数组，表示输入矩阵 (1D 或 2D)。
        l: 正则化参数。
        rho: 超参数。
        k: 当前的标志参数。
        shape: 矩阵的行列数 (rows, cols)。
    
    返回：
        W: 更新后的矩阵。
    """
    # 初始化 W 为 V 的副本
    W = np.copy(V)

    # 计算调整向量 a
    a = np.array([l / rho, -l / rho])
    a1 = np.array([[l / rho, -l / rho]])
    threshold = 2 * (l / rho) ** 2

    # 获取偏移量的索引
    rows, cols = shape
    idx = np.arange(rows * cols).reshape(rows, cols)
    pairs = []

    if k == 0:  # 横向相邻点，满足行减列是 2 的倍数
        for i in range(rows):
            for j in range(cols - 1):
                if (i - j) % 2 == 0:  
                    pairs.append([idx[i, j], idx[i, j + 1]])

    if k == 1:  # 横向相邻点，满足行减列不是 2 的倍数 
        for i in range(rows):
            for j in range(cols - 1):
                if (i - j) % 2 == 1:  
                    pairs.append([idx[i, j], idx[i, j + 1]])
    
    if k == 2:  # 纵向相邻点，满足行减列是 2 的倍数 
        for i in range(rows-1):
            for j in range(cols):
                if (i - j) % 2 == 0:  
                    pairs.append([idx[i, j], idx[i+1, j]])
    if k == 3:  # 纵向相邻点，满足行减列不是 2 的倍数 
        for i in range(rows-1):
            for j in range(cols):
                if (i - j) % 2 == 1:  
                    pairs.append([idx[i, j], idx[i+1, j]])

    pairs = np.array(pairs)
    # horizontal_pairs = (idx[:, :-1].flatten(), idx[:, 1:].flatten())
    # 获取相邻的索引对
    # row_idx, col_idx = horizontal_pairs
    y_values = V[pairs.T]  # 获取对应的 y 值

    # 条件计算，直接对整个数组进行广播操作
    delta_y_positive = y_values - a1.T
    delta_y_negative = y_values + a1.T
    delta_y_zero = y_values - (a @ y_values) / threshold * a1.T

    # 更新规则：通过广播操作和条件判断批量更新
    W[pairs.T] = np.where(
        a @ y_values > threshold,
        delta_y_positive,
        np.where(
            a @ y_values < -threshold,
            delta_y_negative,
            delta_y_zero,
        )
    )

    return W

def step2(V, l, rho, k, shape): #并行版本
    W = np.copy(V)
    flag = [(0,0),(2,1),(1,3),(4,2),(3,4)]
    aim = flag[k]

    rows, cols = shape
    idx = np.arange(rows * cols).reshape(rows, cols)
    a = np.array([-l / rho, -l / rho, -l / rho, -l / rho, 4*l / rho])
    a1 = np.array([[-l / rho, -l / rho, -l / rho, -l / rho, 4*l / rho]])
    threshold = np.linalg.norm(a)**2
    pairs = []
    for m in range(1, rows-1):
        for n in range(1, cols - 1):
            if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
                pairs.append([idx[m, n+1], idx[m+1, n], idx[m, n-1], idx[m-1, n],idx[m,n]])
    pairs = np.array(pairs)
    y_values = V[pairs.T]
    delta_y_positive = y_values - a1.T
    delta_y_negative = y_values + a1.T
    delta_y_zero = y_values - (a @ y_values) / threshold * a1.T

    W[pairs.T] = np.where(
        a @ y_values > threshold,
        delta_y_positive,
        np.where(
            a @ y_values < -threshold,
            delta_y_negative,
            delta_y_zero,
        )
    )

    pairs = []
    a = np.array([-l / rho, -l / rho, -l / rho,  3*l / rho])
    a1 = np.array([[-l / rho, -l / rho, -l / rho,  3*l / rho]])
    m = 0 
    for n in range(1, cols - 1):
        if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
            pairs.append([idx[m, n+1], idx[m+1, n], idx[m, n-1], idx[m, n]])
    m = rows - 1
    for n in range(1, cols - 1):
        if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
            pairs.append([idx[m, n+1], idx[m-1, n], idx[m, n-1], idx[m, n]])
    n = 0
    for m in range(1, rows - 1):
        if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
            pairs.append([idx[m, n+1], idx[m+1, n], idx[m-1, n], idx[m, n]])
    n = cols - 1
    for m in range(1, rows - 1):
        if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
            pairs.append([idx[m, n-1], idx[m+1, n], idx[m-1, n], idx[m, n]])
    pairs = np.array(pairs)
    y_values = V[pairs.T]
    delta_y_positive = y_values - a1.T
    delta_y_negative = y_values + a1.T
    delta_y_zero = y_values - (a @ y_values) / threshold * a1.T

    W[pairs.T] = np.where(
        a @ y_values > threshold,
        delta_y_positive,
        np.where(
            a @ y_values < -threshold,
            delta_y_negative,
            delta_y_zero,
        )
    )

    pairs = []
    a = np.array([-l / rho, -l / rho, 2*l / rho])
    a1 = np.array([[-l / rho, -l / rho, 2*l / rho]])
    m = 0
    n = 0 
    if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
        pairs.append([idx[m, n+1], idx[m+1, n], idx[m, n]])
    m = 0
    n = cols - 1
    if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
        pairs.append([idx[m, n-1], idx[m+1, n], idx[m, n]])
    m = rows - 1
    n = 0 
    if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
        pairs.append([idx[m, n+1], idx[m-1, n], idx[m, n]])
    m = rows - 1
    n = cols - 1
    if  ((2 * m + n) % 5, (m - 2 * n) % 5) == aim:
        pairs.append([idx[m, n-1], idx[m-1, n], idx[m, n]])

    if len(pairs) == 0:
        return W
    pairs = np.array(pairs)
    y_values = V[pairs.T]
    delta_y_positive = y_values - a1.T
    delta_y_negative = y_values + a1.T
    delta_y_zero = y_values - (a @ y_values) / threshold * a1.T

    W[pairs.T] = np.where(
        a @ y_values > threshold,
        delta_y_positive,
        np.where(
            a @ y_values < -threshold,
            delta_y_negative,
            delta_y_zero,
        )
    )


    return W

def dogeADMM(image,rho=1,l=1,B=1,max_iter=100,k=0,threshold = 1e-5, get_loss_seq = False, get_img_seq = False):
    """
    image: The input image to be denoised (2D numpy array).
    rho: Regularization parameter controlling the data fidelity term.
    l:Regularization parameter for controlling sparsity in the solution.
    B: Parameter controlling the grouping strategy for differential operators.
    max_iter: Maximum number of iterations to perform during the optimization process.
    k: A parameter influencing the update rule in differential operators.
    threshold: Convergence threshold to stop the algorithm when the change is smaller than this value.
    get_loss_seq: Boolean flag to return the loss sequence during optimization.
    get_img_seq: Boolean flag to return the sequence of images during the optimization process.
    """

    y= image.flatten(order='C')
    print(f'image shape: {image.shape}')
    Z = copy.deepcopy(y)
    Z_old = np.copy(Z)
    L = generate_L(image)
    D = generate_D(image)
    delta_loss = generate_delta(D,L,k+1)
    delta = generate_delta(D,L,k+1-B)
    img_seq = [y]
    if (k+1-B) % 2 == 0:
        psi = generate_delta(D,L,B)
    else:
        psi = generate_delta(D,L,B-1) @ generate_delta(D,L,1).T
    length = psi.shape[1]
    time0 = time.time()
    Loss = []
    Loss.append(loss(y, Z, delta_loss, l))
    Time = []
    Time.append(time.time()-time0)
    if B == 1:
        alpha = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
        theta = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
        V = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
        for iter in range(max_iter):
            for i in range(4):
                V[i] = delta @ Z - theta[i]
                alpha[i] = step1(V[i],l,rho,i,image.shape) # 这里也可以并行
            
            if k+1-B != 0 :            
                S = copy.deepcopy(y)
                for i in range(4):
                    S = S + rho * delta.T @ (theta[i] + alpha[i])

                Z = np.linalg.inv(np.eye(len(y)) + 4 * rho * delta.T @ delta) @ S     

            else:
                S = copy.deepcopy(y)
                for i in range(4):
                    S = S + rho * (theta[i] + alpha[i])
                Z = S / (1 + 4 * rho)

            for i in range(4):
                theta[i] = theta[i] + alpha[i] - delta @ Z   

            Loss.append(loss(y, Z, delta_loss, l))
            Time.append(time.time()-time0)
            # 终止条件  
            if np.linalg.norm(Z - Z_old) / np.linalg.norm(Z_old) < threshold:
                break
            Z_old = np.copy(Z)
            img_seq.append(Z)

    if B == 2:
        alpha = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
        theta = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
        V = [np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length),np.zeros(length)]
        for iter in range(max_iter):
            for i in range(5):
                V[i] = delta @ Z - theta[i]
                alpha[i] = step2(V[i],l,rho,i,image.shape)
            
            if k+1-B != 0 :            
                S = copy.deepcopy(y)
                for i in range(5):
                    S = S + rho * delta.T @ (theta[i] + alpha[i])

                Z = np.linalg.inv(np.eye(len(y)) + 4 * rho * delta.T @ delta) @ S     

            else:
                S = copy.deepcopy(y)
                for i in range(5):
                    S = S + rho * (theta[i] + alpha[i])
                Z = S / (1 + 5 * rho)


            for i in range(5):
                theta[i] = theta[i] + alpha[i] - delta @ Z

            Loss.append(loss(y, Z, delta_loss, l))
            Time.append(time.time()-time0)
            # 终止条件 
            if np.linalg.norm(Z - Z_old) / np.linalg.norm(Z_old) < threshold:
                break
            Z_old = np.copy(Z)
            img_seq.append(Z)

    print(f'iter:{iter+1}')

    if get_loss_seq:
        if get_img_seq:
                return Z,Loss,Time, img_seq
            
        else:
            return Z,Loss,Time
    else:
        if get_img_seq:
            return Z, img_seq
    
    return Z     



def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

def ADMM(image,k,lambd,rho,max_iter=100,tol=1e-6):
    begin = time.time()
    y = image.flatten(order='C')
    n = y.shape[0]

    L = generate_L(image)
    D = generate_D(image)
    delta = generate_delta(D, L, k+1)

    beta = np.copy(y)

    # z = np.zeros(n)
    u = np.zeros(n)
    beta_old = np.copy(beta)

    if k%2 == 0:
        S=generate_D(image)
        q = k//2
    else:
        S=L
        q = (k+1)//2
    Lq=np.linalg.matrix_power(L, q)
    L2q=np.linalg.matrix_power(L, 2*q)

    z = Lq @ beta
    Loss = []
    Time = []
    Loss.append(loss(y,beta,delta,lambd))
    Time.append(0)

    for i in range(max_iter):
        # Update beta
        beta_old = np.copy(beta)
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
                # 可以保留上一轮的 z 值或者初始化为某个默认值
                # 在此例中保留上一轮的值
                z = z
        else:
            z = soft_thresholding(Lq @ beta - u, lambd / rho)
            
        # Update dual variable u
        u = u - Lq @ beta + z
        Loss.append(loss(y,beta,delta,lambd))
        Time.append(time.time()-begin)


        # Check for convergence
        if np.linalg.norm(beta - beta_old) / np.linalg.norm(beta_old) < tol:
            break


    print("iter:", i)    
    return beta, Loss, Time


'''
FASTTF
'''
def fast_ADMM(image,l,rho,max_iter=20,k=1,tol = 1e-6):
    """
    l ----- lambda
    """
    y = image.flatten(order='C')
    L = generate_L(image)
    D = generate_D(image)
    Z = copy.deepcopy(y)
    Z_old = np.copy(Z)
    delta_loss = generate_delta(D,L,1)
    if k%2==1:
        delta_loss = delta_loss.T
    delta=generate_delta(D, L, k)
    img_seq = [image]
    length = delta.shape[1]

    row = delta.shape[0]
    Loss = []
    Loss.append(loss(y, Z, delta_loss@delta, l))
    Time = []
    begin = time.time()
    Time.append(time.time()-begin)
    
    alpha = delta @ y
    beta= np.copy(y)
    u = np.zeros(row)
    for iter in range(max_iter):
        # beta = np.linalg.inv(np.eye(length) + rho * delta.T @ delta) @ (y+rho*delta.T @ (alpha-u))
        beta_old = np.copy(beta)
        A = np.eye(length) + rho * delta.T @ delta
        b = y + rho * delta.T @ (alpha + u)
        beta = np.linalg.solve(A, b)
        
        # Update alpha
        alpha_value = cp.Variable(row)
        # set objective
        objective = cp.Minimize(0.5 * cp.norm2(delta @ beta - u - alpha_value)**2 + (l/rho) * cp.norm1(delta_loss @ alpha_value))
        # define and solve the problem
        problem = cp.Problem(objective)
        problem.solve()
        alpha = alpha_value.value

        # Update u
        u = u + alpha - delta @ beta
        Loss.append(loss(y, beta, delta_loss@delta, l))
        Time.append(time.time()-begin)

        if np.linalg.norm(delta @ beta - alpha)/np.linalg.norm(beta_old) < tol:
            break
        Z_old = np.copy(Z)
        img_seq.append(Z)
    print("Time execution:", time.time()-begin)    
    return beta,Loss,iter,Time


"""
Original ADMM
"""
def ORIADMM(image,l,rho,max_iter=20,k=1,tol = 1e-6):
    """
    l ----- lambda
    """
    y = image.flatten(order='C')
    L = generate_L(image)
    D = generate_D(image)
    Z = copy.deepcopy(y)
    Z_old = np.copy(Z)
    # delta_loss = generate_delta(D,L,1)
    # if k%2==1:
    #     delta_loss = delta_loss.T
    delta=generate_delta(D, L, k+1)
    img_seq = [image]
    length = delta.shape[1]

    row = delta.shape[0]
    Loss = []
    Loss.append(loss(y, Z, delta, l))
    print("Loss:", Loss[0])
    Time = []
    begin = time.time()
    Time.append(time.time()-begin)
    
    alpha = delta @ y*0.5
    beta= np.copy(y)
    u = np.zeros(row)
    for iter in range(max_iter):
        # beta = np.linalg.inv(np.eye(length) + rho * delta.T @ delta) @ (y+rho*delta.T @ (alpha-u))
        beta_old = np.copy(beta)
        A = np.eye(length) + rho * delta.T @ delta
        b = y + rho * delta.T @ (alpha + u)
        beta = np.linalg.solve(A, b)
        
        # Update alpha
        alpha = soft_thresholding(delta @ beta - u, l / rho)

        # Update u
        u = u + alpha - delta @ beta
        Loss.append(loss(y, beta, delta, l))
        Time.append(time.time()-begin)

        if np.linalg.norm(delta @ beta - alpha)/np.linalg.norm(beta_old) < tol:
            break
        Z_old = np.copy(Z)
        img_seq.append(Z)
    print("Time execution:", time.time()-begin)    
    return beta,Loss,iter,Time
