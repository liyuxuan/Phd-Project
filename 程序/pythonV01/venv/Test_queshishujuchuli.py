import numpy as np
from numpy import nan
# 2.2以后已经进行改写;
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

#承装的容器为链表;
X=np.array(
    [[nan,0,3],
    [2,9,-8],
    [1,nan,1],
    [5,2,4],
    [7,6,-3]]
)

# 进行均值内容的编辑;
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)

X2 =imputer.fit_transform(X)

# print(X2)

# 可以通过计算第一列（不计算第一个元素X[0,0]）的均值来检查这个数学结果;并将计算值与矩阵的第一个元素X2[0,0]进行对比;
out1,out2=np.mean(X[1:,0]),X2[0,0]
print(out1,out2)

######################################################################################
# 进行均值的内容编辑;
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median',verbose=0)

X3 =imputer.fit_transform(X)
print(X3)

#这次计算种植不包括X[0,0]，并将结果与X3[0,0]进行对比
out3,out4=np.median(X[1:,0]),X3[0,0]
print(out3,out4)

