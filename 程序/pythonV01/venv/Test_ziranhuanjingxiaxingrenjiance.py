import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
# # 使用循环来读取数据;
# for i in range(5):
#     filename="figures/per0000%d.png"%(i+1)
#     img=cv2.imread(filename)
#     plt.subplot(1,5,i+1)
#     plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#     plt.axis("off")

win_size=(48,96)
block_size=(16,16)
block_stride=(8,8)
cell_size=(8,8)
num_bins=9
# 此函数，实现HOG描述符所需要的值。影响最大的参数是窗口大小(win_size)
hog=cv2.HOGDescriptor(win_size,block_size,block_stride,cell_size,num_bins)
# 构建的正样本数据集;
X_pos=[]
for i in range(5):
    filename="figures/per0000%d.png"%(i+1)
    img=cv2.imread(filename)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if img is None:
        print("Could not find image %s"%filename)
        continue
    X_pos.append(hog.compute(img,(64,64)))

X_pos=np.array(X_pos,dtype=np.float32)
y_pos=np.ones(X_pos.shape[0],dtype=np.int32)
# 一共挑选了5张图，其中每张图共有1980个特征值点;
print(X_pos.shape,y_pos.shape)
# 寻找行人图像的对立图像的最佳方法是收集一个数据集，其中的图像类似正样本，但不包含行人。可以包括除了行人以外的其他东西。

import os
hroi=128
wroi=67
X_neg=[]
for j in range(5):
    filename="figures/par%d.jpg"%(j+1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(512,512))
    for t in range(5):
        rand_y=random.randint(0,img.shape[0]-hroi)
        rand_x=random.randint(0,img.shape[1]-wroi)
        roi=img[rand_y:rand_y+hroi,rand_x:rand_x+wroi,:]
        X_neg.append(hog.compute(roi,(64,64)))

# 确保都是浮点数;
X_neg=np.array(X_neg,dtype=np.float32)
y_neg=-np.ones(X_neg.shape[0],dtype=np.int32)

# 使用scikit-learn中分离出来的train_test_split函数处理;
X=np.concatenate((X_pos,X_neg))
y=np.concatenate((y_pos,y_neg))
from sklearn import model_selection as ms
X_train,X_test,y_train,y_test=ms.train_test_split(X,y,test_size=0.2,random_state=42)
# 确保不含有行人的图像作为背景负样本;

def train_svm(X_train,y_train):
    svm=cv2.ml.SVM_create()
    svm.train(X_train,cv2.ml.ROW_SAMPLE,y_train)
    return svm

def score_svm(svm,X,y):
    from sklearn import metrics
    _,y_pred=svm.predict(X)
    return metrics.accuracy_score(y,y_pred)

# 使用两个端的函数调用来训练和评估SVM
svm=train_svm(X_train,y_train)
out1=score_svm(svm,X_train,y_train)
out2=score_svm(svm,X_test,y_test)
print(out1)
print(out2)
# plt.show()
# 由于使用HOG特征描述子，训练数据集合时没有出现错误。模型在在数据集上过拟合。
# 训练数据集比测试数据集表现好，这种情况表明模型已经采用记住训练样本的方法，而不是试图从中抽象出一个有意义的决策规则。

# 提升模型性能，可以使用自举，通常使用HOG+SVM特征进行行人检测;
# 其思路为在训练数据机上训练完SVM后，对模型进行评分，发现模型得到一些假正的结果。假正:预测为正样本(+)，实际负样本(-),
# 在场景中，表明SVM错误认为图像中包含行人，因此将有问题的图片添加到训练数据集中，并使用这些额外有问题的推向重新训练SVM
# ，这样算法就可以学到如何把这个图像正确分类。一直重复直到最优结果;

# 这样的过程最多3次;
# 1.训练并且评估模型;
score_train=[]
score_test=[]
for s in range(3):
    svm=train_svm(X_train,y_train)
    score_train.append(score_svm(svm,X_train,y_train))
    score_test.append(score_svm(svm, X_test, y_test))
# 2.从测试数据集中找到假正的图片。如果没有的话，完成训练;
    _,y_pred=svm.predict(X_test)
    false_pos=np.logical_and((y_test.ravel()==-1),(y_pred.ravel()==1))

    if not np.any(false_pos):
        print("no more false positives:done")
        break
# 3.把假正的图片添加到训练数据集中，然后重复训练过程;
    X_train=np.concatenate((X_train,X_test[false_pos,:]),axis=0)
    y_train = np.concatenate((y_train, y_test[false_pos, :]), axis=0)

print(score_train)
print(score_test)