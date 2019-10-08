from sklearn import preprocessing
import numpy as np
X=np.array([[1.,-2.,2.],
            [3.,0.,0.],
            [0.,1.,-1.]])
# print(X)

# 对使用的矩阵进行标准化;
X_scaled=preprocessing.scale(X)
# print(X_scaled)
# 检查均值和方差验证缩放后的数据矩阵X_scaled，确实已经完成了标准化操作。一个标准化后的特征矩阵应该每行都接近1.
out1=X_scaled.std(axis=0)

# print(out1)
X_normalize_l1=preprocessing.normalize(X,norm='l1')
# print(X_normalize_l1)
# L2范数缩放;
X_normalize_l2=preprocessing.normalize(X,norm='l2')
# print(X_normalize_l2)

min_max_scaler=preprocessing.MinMaxScaler()
X_min_max=min_max_scaler.fit_transform(X)
# 默认的情况下，数据会缩放到0~1之间。
# print(X_min_max)

# 可以通过传入MinMaxScaler构造函数一个关键字参数feature_range指定不同的范围:
min_max_scaler=preprocessing.MinMaxScaler(feature_range=(-10,10))
X_min_max2=min_max_scaler.fit_transform(X)
# print(X_min_max2)

# print(X)
#二值化，是使用特征值的标准来定义阈值，
binarizer=preprocessing.Binarizer(threshold=0.5)
X_binarized=binarizer.transform(X)
print(X_binarized)

