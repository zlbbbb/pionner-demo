import numpy as np
import math

#action a1 上 a2 下 a3 左 a4 右 at 原地
#rb=rf=-1  rt=1 gamma=0.9
q_table=np.array([
    [-1,-1,0,-1,0],
    [-1,-1,1,0,1],
    [0,1,-1,-1,0],
    [-1,-1,-1,0,1],
])

def q_table_v_matrix_methods(v):
    #转化为计算q的状态值矩阵
    v=v.reshape(-1) #展成一维的只有一列的行向量

    #状态转移矩阵
    transfer_matrix=np.array(
        [   #索引从0开始  0表示s1 1表示s2 2表示s3 3表示s4
            [0,1,2,0,0],
            [1,1,3,0,1],
            [0,3,2,2,2],
            [1,3,3,2,3],
        ]
    )
    q_table_v_matrix=transfer_matrix.copy()
    for i in range(transfer_matrix.shape[0]):
        for j in range(transfer_matrix.shape[1]):
            q_table_v_matrix[i,j]=v[transfer_matrix[i,j]]

    return [q_table_v_matrix,transfer_matrix]

def value_iteration(q_table,gamma,q_table_v_matrix_methods):
    #第一步,q_table给定状态值  回报R矩阵
    n=q_table.shape[0] #状态数量
    v_new=np.random.rand(n).reshape(-1,1) #初始化v_new 随机生成n行一列的列向量
    v=np.array([math.inf for i in range(n)]).reshape(-1,1) #初始化v 全部赋值为无穷大
    #开始迭代
    while abs(np.sum(v_new-v))>1e-28:
        #收缩映射判定终止条件
        v=v_new

        #获得位置转移矩阵和q_table计算的v矩阵
        q_table_v_matrix,transfer_matrix=q_table_v_matrix_methods(v)

        #第二步计算q_value   r+gamma*v(s')
        q_value=q_table+gamma*q_table_v_matrix

        #第三步选择最优策略  pi k+1 (a|s) 得到每个状态s的策略
        pi=np.argmax(q_value,axis=1).reshape(-1,1) #axis沿轴操作 1表示沿行操作 找到每行的最大值索引 一维列向量
        #axis=0表示沿列操作 找到每列的最大值索引     axis 0表示第一维度 1表示第二维度
        #根据最优策略计算
        #选取qk(s,a)最大的值     例如s1 选取 a2动作   得到q_table[0,1]
        R=[q_table[idx,pi[idx]] for idx in range(pi.shape[0])]

        #第四步计算概率P
        P=np.zeros((n,n))
        for idx in range(pi.shape[0]):
            P[idx,transfer_matrix[idx,pi[idx]]]=1

        #第五步更新v_new
        v_new=R+(gamma*P).dot(v)

    return v_new

result=value_iteration(q_table,0.9,q_table_v_matrix_methods)
print(result)

