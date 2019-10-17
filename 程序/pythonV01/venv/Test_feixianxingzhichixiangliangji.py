import cv2
import numpy as np
# 数据源
from sklearn import datasets
from sklearn import model_selection as ms
from sklearn import metrics
import matplotlib.pyplot as plt

X,y=datasets.make_classification(n_samples=100,n_features=2,n_redundant=0,n_classes=2,random_state=7816)

X=X.astype(np.float32)
y=y*2-1
X_train,X_test,y_train,y_test=ms.train_test_split(X,y,test_size=0.2,random_state=42)

kernels=[cv2.ml.SVM_LINEAR,
         cv2.ml.SVM_INTER,
         cv2.ml.SVM_SIGMOID,
         cv2.ml.SVM_RBF
         ]

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


# 从kernels列表中提取一个核，并把它传入到SVM类的setKernels方法中。
for idx,kernel in enumerate(kernels):
    #1.创建SVM并设置kernel方法:
    svm=cv2.ml.SVM_create()
    svm.setKernel(kernel)

    #2.训练分类器;
    svm.train(X_train,cv2.ml.ROW_SAMPLE,y_train)
    # 3.使用前面引入的scikit-learn的度量模块对模型打分;
    _,y_pred=svm.predict(X_test)
    accuracy=metrics.accuracy_score(y_test,y_pred)

    # 4.以2x2的形式绘制子图画出决策边界;
    plt.subplot(2,2,idx+1)
    plot_decision_boundary(svm,X_test,y_test)
    plt.title("accuracy=%.2f"%accuracy)


plt.show()
