
# 数据生成;
# 2分类具有2个不同的目标标签(n_classes=2).为了简化的目的，限制他仅有2个特征值(n_features=2)
from sklearn import datasets
# n_samples=100:创建100个数据样本;n_features=2:表示两个特征值;n_classes=2:表示数据分为2类;
X,y=datasets.make_classification(n_samples=100,n_features=2,n_redundant=0,n_classes=2,random_state=7816)

#结果中，X有100行（数据样本）和2列（特征），而向量y应该有包含所有目标标签的单独1列。
# print(X)
# print(X.shape)
# print(y)
# print(y.shape)

# 数据集可视化;
import matplotlib.pyplot as plt
# x值X[:,0],y值[:,1],目标标签作为色彩值传入(c=y)

plt.scatter(X[:,0],X[:,1],c=y,s=100)

# plt.xlabel("x values")
# plt.ylabel("y values")
# # 如图，存在2类数据交叉在一起。导致很难正确分类;
# plt.show()

# 数据集预处理
# 把数据点划分为训练集和测试集。
# 进行相应的OpenCV数据的处理之前，需要进行的准备工作;
# 1.X中所有的特征一定要32位浮点值
# 2.目标标签一定要么-1，要么+1

import numpy as np
X=X.astype(np.float32)
# print(X)

y=y*2-1
# print(y)
# 将数据传到scikit-learn的train_test_split函数中;
from sklearn import model_selection as ms
# 根据自身要求调节测试比例;20%
X_train,X_test,y_train,y_test=ms.train_test_split(X,y,test_size=0.2,random_state=42)

# 构建支持向量机
# 在OpenCV中，SVM的构建、训练和测试得分的方式与之前的各个学习算法的方式相同；
# 1.调用create方法创建一个新的SVM
import cv2
svm=cv2.ml.SVM_create()

out1=svm.setKernel(cv2.ml.SVM_LINEAR)

# print(out1)
# 2.调用分类器的train方法来找到最优决策边界;
svm.train(X_train,cv2.ml.ROW_SAMPLE,y_train)

# 3.调用分类器的predict方法来预测测试数据集中所有数据样本的目标标签;
_,y_pred=svm.predict(X_test)
# print(_)
# print(y_pred)

# 4.使用scikit-learn中的metrics模块对分类器打分：
from sklearn import metrics
out2=metrics.accuracy_score(y_test,y_pred)
print(out2)

# 决策边界可视化
def plot_decision_boundary(svm,X_test,y_test):
    # 为了生成方格(也叫作网格)，首先需要知道数据样本在x-y平面上所占用的空间大小。为了找到平面上最左边的点，在X_test
    # 中找到最大的x值。我们不想有任何数据点落在边界上，因此对边缘+1或者-1
    x_min,x_max = X_test[:,0].min()-1,X_test[:,0].max()+1
    y_min,y_max = X_test[:,1].min() - 1, X_test[:, 1].max() + 1
    # 通过这些边界的值，创建一个合适的网格;
    h=0.02
    # 使用Numpy的arrage(start,end,step)函数来创建一个线性空间，其值在start和stop之间，按照step步长进行分割;
    xx,yy       = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    X_hypo      = np.c_[xx.ravel().astype(np.float32),yy.ravel().astype(np.float32)]

    #  其中，将这些值转换为32位浮点数值！否则，OpenCV将无法正确执行。得到的结果目标标签zz被用来创建一个特征结构的颜色映射;
    _,zz        = svm.predict(X_hypo)

    zz          = zz.reshape(xx.shape)
    plt.contour(xx,yy,zz,cmap=plt.cm.coolwarm,alpha=0.8)
    plt.scatter(X_test[:,0],X_test[:,1],c=y_test,s=200)

plot_decision_boundary(svm,X_test,y_test)
plt.show()






