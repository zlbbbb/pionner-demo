import numpy as np
def iteractive_solution(n_iter,R,P,gamma):
    ''' v_pi初始化
    :param n_iter: 迭代次数
    :param R:
    :param P:
    :param gamma:
    :return: v_pi
    '''
    n=R.shape[0]
    v=np.random.rand(n,1)

    for iter in range(n_iter):
        v=R+(gamma*P).dot(v)
    return v
R=np.array([(0.5*0+0.5*(-1)),1.,1.,1.]).reshape(-1,1)
P=np.array(
    [[0,0.5,0.5,0],
     [0,0,0,1],
     [0,0,0,1],
     [0,0,0,1]]
)
result=iteractive_solution(n_iter=100,R=R,P=P,gamma=0.9)
print(result)