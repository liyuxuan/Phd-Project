import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model

boston=datasets.load_boston()

#(1)载入对象来替换初始化的命令;
# linreg=linear_model.LinearRegression()
# print(linreg)

# 对于Lasso回归算法,使用下面代码替换如上代码(1)
# lassoreg=linear_model.Lasso()
# print(lassoreg)

#
ridger=linear_model.Ridge()
print(ridger)
