from sklearn import datasets
import numpy as np

# 载入数据集合;
data=datasets.load_breast_cancer()
# 所有的数据都存在于2维的特征矩阵data.data中,其中行表示的是数据样本,列表示的是特征值;

out1=data.data.shape
# print(out1)

# 特征名称的显示;
out2=data.feature_names
# print(out2)
# 二分类任务，希望中找到两个目标名称;
out3=data.target_names
# print(out3)

#保留数据样本，20%的数据用于测试;
import sklearn.model_selection as ms

# 获取目标处理的内容;
data_pre=data.data
target=data.target

# 将数据进行浮点类型转换;
data_pre=np.array(data_pre,dtype=np.float32)
target=np.array(target,dtype=np.float32)

X_train,X_test,y_train,y_test=ms.train_test_split(data_pre,target,test_size=0.2,random_state=42)

print(X_train.shape,X_test.shape)

# 构建决策树;
from sklearn import tree
dtc=tree.DecisionTreeClassifier()

# 训练决策树;
out4=dtc.fit(X_train,y_train)
# print(out4)

# 由于没有指定预剪枝参数，预测的决策树巨大，训练数据集合上会有完美得分;
out5=dtc.score(X_train,y_train)
# print(out5)

out6=dtc.score(X_test,y_test)
# print(out6)
import matplotlib.pyplot as plt

# # 使用不同的max_depth值来重复构建不同的决策树;
# max_depths=np.array([1,2,3,5,7,9,11])
# # 对于上边的每一个值，我从开始到结束运行完整的模型级联，同时记录在训练集和测试集上的得分,使用for循环来完成;
# train_score=[]
# test_score=[]
# for d in max_depths:
#     dtc=tree.DecisionTreeClassifier(max_depth=d)
#     dtc.fit(X_train,y_train)
#     train_score.append(dtc.score(X_train,y_train))
#     test_score.append(dtc.score(X_test,y_test))
#
# # 每一个max_depths内的值都构建一个决策树，并且在数据上训练决策树，建立所有在训练数据集和测试集上的得分列表。可以使用matlab绘画出深度和得分之间的函数关系图;

# plt.style.use("ggplot")
#
# plt.figure(figsize=(10,6))
# plt.plot(max_depths,train_score,"o-",linewidth=3,label="train")
# plt.plot(max_depths,test_score,"s-",linewidth=3,label="test")
# #######################################################################################################################
# 设置将节点称为一个叶节点所需要的最小样本数；
train_score=[]
test_score=[]
min_samples=np.array([2,4,8,16,32])
for s in min_samples:
    dtc=tree.DecisionTreeClassifier(min_samples_leaf=s)
    dtc.fit(X_train,y_train)
    train_score.append(dtc.score(X_train,y_train))
    test_score.append(dtc.score(X_test,y_test))

plt.figure(figsize=(10,6))
plt.plot(min_samples,train_score,"o-",linewidth=3,label="train")
plt.plot(min_samples,test_score,"s-",linewidth=3,label="test")

plt.xlabel("max_depth")
plt.ylabel("score")

plt.legend()
plt.show()






