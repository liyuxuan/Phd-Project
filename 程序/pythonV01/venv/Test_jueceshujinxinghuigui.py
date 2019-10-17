import numpy as np
# 想要使用一个决策树来你和sin曲线.使用随机数生成器对数据点添加一些噪声;
rng=np.random.RandomState(42)

X=np.sort(5*rng.rand(100,1),axis=0)
y=np.sin(X).ravel()
# 在y中的每一个间隔添加噪声y[::2],将尺度缩放0.5倍，不会引入过多抖动;
y[::2]+=0.5*(0.5-rng.rand(50))
# 构建回归树，轻微的不同之处是分割标准gini和entropy无法应用到回归任务中，作为替代，scikit-learn提供了2个不同的分割标准:

from sklearn import tree
regr1=tree.DecisionTreeRegressor(max_depth=2,random_state=42)
out1=regr1.fit(X,y)
print(out1)

regr2=tree.DecisionTreeRegressor(max_depth=5,random_state=42)
out2=regr2.fit(X,y)

X_test=np.arange(0.0,5.0,0.01)[:,np.newaxis]

y_1=regr1.predict(X_test)
y_2=regr2.predict(X_test)

import matplotlib.pyplot as plt
plt.style.use("ggplot")

plt.scatter(X,y,c="k",s=50,label="data")
plt.plot(X_test,y_1,label="max_depth=2",linewidth=5)
plt.plot(X_test,y_2,label="max_depth=5",linewidth=3)

plt.xlabel("data")
plt.ylabel("target")

plt.legend()
plt.show()