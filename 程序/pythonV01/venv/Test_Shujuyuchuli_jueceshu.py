import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
import numpy as np
plt.style.use("ggplot")

# 加载数据信息内容;
data=[
    {'age': 33, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.66, 'K': 0.06, 'drug': 'A'},
    {'age': 77, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.19, 'K': 0.03, 'drug': 'D'},
    {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.80, 'K': 0.05, 'drug': 'B'},
    {'age': 39, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.19, 'K': 0.02, 'drug': 'C'},
    {'age': 43, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'high', 'Na': 0.36, 'K': 0.03, 'drug': 'D'},
    {'age': 82, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.09, 'K': 0.09, 'drug': 'C'},
    {'age': 40, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.89, 'K': 0.02, 'drug': 'A'},
    {'age': 88, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.80, 'K': 0.05, 'drug': 'B'},
    {'age': 29, 'sex': 'F', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.35, 'K': 0.04, 'drug': 'D'},
    {'age': 53, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.54, 'K': 0.06, 'drug': 'C'},
    {'age': 36, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.53, 'K': 0.05, 'drug': 'A'},
    {'age': 63, 'sex': 'M', 'BP': 'low', 'cholesterol': 'high', 'Na': 0.86, 'K': 0.09, 'drug': 'B'},
    {'age': 60, 'sex': 'M', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.66, 'K': 0.04, 'drug': 'C'},
    {'age': 55, 'sex': 'M', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.82, 'K': 0.04, 'drug': 'B'},
    {'age': 35, 'sex': 'F', 'BP': 'normal', 'cholesterol': 'high', 'Na': 0.27, 'K': 0.03, 'drug': 'D'},
    {'age': 23, 'sex': 'F', 'BP': 'high', 'cholesterol': 'high', 'Na': 0.55, 'K': 0.08, 'drug': 'A'},
    {'age': 49, 'sex': 'F', 'BP': 'low', 'cholesterol': 'normal', 'Na': 0.27, 'K': 0.05, 'drug': 'C'},
    {'age': 27, 'sex': 'M', 'BP': 'normal', 'cholesterol': 'normal', 'Na': 0.77, 'K': 0.02, 'drug': 'B'},
    {'age': 51, 'sex': 'F', 'BP': 'low', 'cholesterol': 'high', 'Na': 0.20, 'K': 0.02, 'drug': 'D'},
    {'age': 38, 'sex': 'M', 'BP': 'high', 'cholesterol': 'normal', 'Na': 0.78, 'K': 0.05, 'drug': 'A'}
]
target=[d["drug"] for d in data]
target=[ord(t)-65 for t in target]

# print(target)
# 将所有特征转换为数值特征;
vec=DictVectorizer(sparse=False)

# 把想要转换的数据集传入到fit_transform方法中;
data_pre=vec.fit_transform(data)

# 确保数据变量兼容OpenCV，将所有值转换为浮点类型;
data_pre= np.array(data_pre,dtype=np.float32)
target  = np.array(target,dtype=np.float32)

# print(target)

# 数据集合转换为训练集合测试集合;
import sklearn.model_selection as ms
# test_size=1为测试量；
X_train,X_test,y_train,y_test=ms.train_test_split(data_pre,target,test_size=5,random_state=42)

import  cv2
# 构建决策树;
dtree   =   cv2.ml.DTrees_create()

# 在训练数据集上使用决策树，使用train方法;
out4    =   dtree.train(X_train,cv2.ml.ROW_SAMPLE,y_train)
# 指定X_train中的数据样本是
# 以行的方式(使用cv2.ml.ROW_SAMPLE)
# 以列的方式(cv2.ml.COL_SAMPLE)
y_pred  =   dtree.predict(X_test)

from sklearn import metrics
from sklearn import tree
out3    =   metrics.accuracy_score(y_test,dtree.predict(X_test))
out4    =   metrics.accuracy_score(y_train,dtree.predict(X_train))
print(out3)

#
# # 可视化训练得到的决策树
# dtc = tree.DecisionTreeClassifier()          # 创建空的决策树  采用sklearn方法
# dtc.fit(X_train, y_train)                    # 在训练数据集上训练决策树
# print (dtc.score(X_train, y_train))
# print (dtc.score(X_test , y_test) )






