import numpy as np
import cv2
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt

plt.style.use("ggplot")
# 进行数据集合的载入;
iris=datasets.load_iris()
# print(iris)
shape=iris.data.shape
# print(shape)

#相应的特征值包括几个相应的内容;
features=iris.feature_names
# print(features)

# 每个数据点都有一个类别标签存储在target中;
out1=iris.target.shape
# print(out1)

out2=np.unique(iris.target)
# print(out2)

#简化二分类的方法;
idx=iris.target!=2
data=iris.data[idx].astype(np.float32)
target=iris.target[idx].astype(np.float32)
plt.scatter(data[:,0],data[:,1],c=target,cmap=plt.cm.Paired,s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# plt.show()
#发数据分为训练集合测试集;
X_train,X_test,y_train,y_test=model_selection.train_test_split(data,target,test_size=0.1,random_state=42)

out3,out4=X_train.shape,y_train.shape
# print(out3)
# print(out4)
out5,out6=X_test.shape,y_test.shape
# print(out5)
# print(out6)

#训练分类器
#创建逻辑回归分类器与创建一个k-nn分类器的步骤基本一致;
lr=cv2.ml.LogisticRegression_create()
# print(lr)

#指定期望的训练方法;
lr.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
lr.setMiniBatchSize(1)

#指定迭代的次数;
lr.setIterations(100)
out7=lr.train(X_train,cv2.ml.ROW_SAMPLE,y_train)
# print(out7)

#算法需要添加一个额外的权重用于设置补偿或者偏差

#检索权重;
out8=lr.get_learnt_thetas()
# print(out8)

# ret,y_pred=lr.predict(X_train)
# out9=metrics.accuracy_score(y_train,y_pred)
# print(out9)

#只表明模型可以记住训练模型，并不意味着能够正确分类一个新的、未知的数据点;
ret,y_pred=lr.predict(X_test)
out10=metrics.accuracy_score(y_test,y_pred)
print(out10)
