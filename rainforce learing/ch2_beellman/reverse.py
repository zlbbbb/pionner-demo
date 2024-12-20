import numpy as np
#贝尔曼公示状态求解
def closed_from_solution(R,P,gamma):
    #行号
    n=R.shape[0]
    #单位阵
    I=np.identity(n)
    matrix_inverse=np.linalg.inv(I-gamma*P)

    #矩阵点乘
    return matrix_inverse.dot(R)

R=np.array([(0.5*0+0.5*(-1)),1.,1.,1.]).reshape(-1,1)
P=np.array(
    [[0,0.5,0.5,0],
     [0,0,0,1],
     [0,0,0,1],
     [0,0,0,1]]
)
result=closed_from_solution(R,P,0.9)
print(result)