import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model

import matplotlib.pyplot as plt
plt.style.use('ggplot')

boston=datasets.load_boston();
# print(boston)

#data:所有数据的集合;
#feature_names:所有特征的名字;
#target:所有的目标值;
#descr获得更多数据集的信息;

linreg=linear_model.LinearRegression()

# print(linreg)
X_train,X_test,y_train,y_test=modsel.train_test_split(boston.data,boston.target,test_size=0.1,random_state=42)

#其中，scikit-learn中的train函数叫做fit，其他操作和OpenCV是完全一样的.
out=linreg.fit(X_train,y_train)
# print(out)

#预测真实的均方差;
out2=metrics.mean_squared_error(y_train,linreg.predict(X_train))
# print(out2)

#Linreg对象的Score方法返回的是确定系数(R方值)
out3=linreg.score(X_train,y_train)
# print(out3)

#测试模型
y_pred=linreg.predict(X_test)
out4=metrics.mean_squared_error(y_test,y_pred)
# print(out4)

# 进行数据的描绘;
# plt.figure(figsize=(10,6))
#
# plt.plot(y_test,linewidth=3,label='ground truth')
# plt.plot(y_pred,linewidth=3,label='predicted')
# plt.legend(loc="best")
# plt.xlabel("test data points")
# plt.ylabel("target value")

# plt.show()
#模型在那些真实的房价非常高或者非常低的地方误差很大.
#形式化数据方差的数量，这样就可以通过计算R方值解释现象;

plt.plot(y_test,y_pred,'o')
plt.plot([-10,60],[-10,60],'k--')
plt.axis([-10,60,-10,60])
plt.xlabel("ground truth")
plt.ylabel("predicted")

scorestr=r'R$^2$=%.3f'%linreg.score(X_test,y_test)
errstr='MSE=%.3f'%metrics.mean_squared_error(y_test,y_pred)
plt.text(-5,50,scorestr,fontsize=12)
plt.text(-5,45,errstr,fontsize=12)
#对于良好的数据点，都应该分布在对角线上，因为y_pred总是和y_true相等。与对角线存在偏差表明模型预测出现的误差和方差是模型无法解释的。
#R^2解释离散度;均方误差。这些用来评估复杂模型的硬指标;

plt.show()

