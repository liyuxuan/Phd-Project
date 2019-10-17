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
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

from sklearn import tree
# dtc     =   tree.DecisionTreeClassifier()
dtc=tree.DecisionTreeClassifier(criterion="entropy")
# print(dtc)
# 使用fit方法来训练决策树;
out1    =   dtc.fit(X_train,y_train)
# print(dtc.fit(X_train,y_train))
out2    =   dtc.score(X_train,y_train)
out3    =   dtc.score(X_test,y_test)
# print(out2)
# print(out3)

out4=dtc.feature_importances_
print(out4)
# range(14):特征类别的长度;
plt.barh(range(14),dtc.feature_importances_,align="center",tick_label=vec.get_feature_names())



plt.show()




