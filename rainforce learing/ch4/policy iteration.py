import numpy as np
import math
#action a1 上 a2 右 a3 下 a4 左 at 原地
#rb=rf=-1  rt=1 gamma=0.9
#r矩阵
q_table_r=np.array(
    [
        [-1,-1,0,-1,0],
        [-1,-1,1,0,-1],
        [0,1,-1,-1,0],
        [-1,-1,-1,0,1],
    ]
)

#状态转移矩阵
q_table_transfer_matrix=np.array(
    [ #索引下标从0开始  状态s采取action转移到的状态
        [0,1,2,0,0],
        [1,1,3,0,1],
        [0,3,2,2,2],
        [1,3,3,2,3],
    ]
)
def q_table_v_matrix_methods(v,transfer_matrix):
    #计算q的状态值矩阵
    v=v.reshape(-1) #转化为一维n行一列向量

    #状态转移矩阵副本
    q_table_v_matrix=transfer_matrix.copy()

    #将状态值放置到q_table_v_matrix中
    for i in range(transfer_matrix.shape[0]):
        for j in range(transfer_matrix.shape[1]):
            q_table_v_matrix[i,j]=v[transfer_matrix[i,j]]  #q_value内先放v  state value  q-v

    return q_table_v_matrix

#PE
def iterative_solution(R,P,gamma):
    #n_iter 迭代次数 作为truncated的条件
    n=R.shape[0]
    v_new=np.random.rand(n).reshape(-1,1) #生成随机初始化v
    v=np.array([math.inf for i in range(n)]).reshape(-1,1) #展成列向量
    #iteration条件
    #truncated在这里截断 若一轮为值迭代 无穷轮为策略迭代 具体条件可更改
    while abs(np.sum(v_new - v)) > 1e-8:  #迭代到收敛
    #for i in range(80):#轮数小时  不一定收敛
        #贝尔曼公式计算
        v=v_new
        v_new=R+(gamma*P).dot(v)

    return v_new

#策略迭代
def policy_iteration(q_table,transfer_matrix,gamma,q_table_v_matrix_methods):
    #给定初始化策略
    #第一步 # 行为状态s  列为动作action
    action_n=q_table.shape[1]
    state_n=q_table.shape[0]
    #给定策略
    pi=np.random.randint(low=0,high=action_n,size=state_n) #生成随机策略 从s1开始到结束

    #状态值收敛时停止
    v=np.array([0 for i in range(state_n)]).reshape(-1,1) #定义v
    v_new=np.array([math.inf for i in range(state_n)]).reshape(-1,1)

    while abs(np.sum(v_new-v))>1e-8:
        v=v_new #贝尔曼公式更新
        #第二步计算状态值
        #计算R  q_table奖励矩阵 状态采取行动 对应i,j位置  idx为状态 pi[idx]为动作
        R=np.array([q_table[idx, pi[idx]] for idx in range(pi.shape[0])])

        #计算P 状态idx下 采取策略pi[idx]转移到的状态，设置为1
        P=np.zeros((state_n,state_n))
        for idx in range(pi.shape[0]):
            P[idx,transfer_matrix[idx,pi[idx]]]=1 #采取的策略为1
        #P概率矩阵  P pi k
        #计算该策略下的状态值v_new R确定 P确定 gamma确定
        v_new=iterative_solution(R,P,gamma)

        #第三步 由该状态值选择最优策略
        #计算q_table   r+gamma*vk  transfer_matrix由网格世界限制决定  Q表中的v矩阵更新
        q_table_v_matrix=q_table_v_matrix_methods(v_new,transfer_matrix)
        #计算q_value 接着计算Q值 Q(s,a)=Q表(s,a)+gamma*V矩阵(s,a)
        q_value=q_table+gamma*q_table_v_matrix

        #选择最优策略
        pi=np.argmax(q_value,axis=1).reshape(-1,1) #此使更新为pi k+1
    return pi

result=policy_iteration(q_table_r,q_table_transfer_matrix,0.9,q_table_v_matrix_methods)
print(result)

