import numpy as np
x=np.linspace(0,10,100)
# print(x)
# 真实的数据总会包含噪音。为了遵守这一事实，我们也在目标值y_true上边添加噪声。通过sin函数上加噪声来完成这个操作：
# print(np.sin(x))
# print(np.random.rand(x.size))
y_true=np.sin(x)+np.random.rand(x.size)-0.5
# print(y_true)
y_pred=np.sin(x)

import matplotlib.pyplot as plt
plt.style.use("ggplot")

plt.plot(x,y_pred,linewidth=4,label='model')
plt.plot(x,y_true,'o',label='data')

plt.xlabel('x')
plt.ylabel('y')

plt.legend(loc='lower left')
plt.savefig("figures/02.06-sine.png")
# plt.imshow()
# plt.show()

# 用来判断模型预测的优势最好的方法是均方误差，对于每个数据点，计算预测值和每个实际y值的差值，然后对他平方，然后计算这个平方误差在所有点上的均值
mse=np.mean((y_true-y_pred)**2)
print(mse)
# scikit-learn为其进行进行方便，实现均方误差的实现；
from sklearn import metrics
met=metrics.mean_squared_error(y_true,y_pred)
print(met)

#另一个常用的指标是计算数据的离散度或方差：如果每个数据点都等于所有数据点的均值，那么这些数据就没有离散度和方差，我们就可以用一个值来预测所有未来的数据点的值。
r2=1.0-mse/np.var(y_true)
print(r2)

# 利用scikit也可以完成相同的效果;
r3=metrics.r2_score(y_true,y_pred)
print(r3)

