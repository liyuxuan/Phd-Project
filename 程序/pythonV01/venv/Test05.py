import numpy as np
np.random.seed(42)

y_true=np.random.randint(0,2,size=5)
print(y_true)
# # 这两类有时称为正样本（所有标签是1的数据点）以及负样本（所有标签是0的数据点）
# # 假设有一个尝试预测前面的类的类标签的分类器，假设并不准确，则每次都是设置为1，通过硬编码预测的标签来模拟行为：
y_pred=np.ones(5,dtype=np.int32)
print(y_pred)
# # 准确率统计的是测试数据集中那些被预测正确的点的数量除以测试数据集的大小。
#
# a=np.sum(y_true==y_pred)/len(y_true)
# print(a)
#
# # 更好的方法是使用scikit-learn中的metrics：
from sklearn import metrics
# b=metrics.accuracy_score(y_true,y_pred)
# print(b)
#
# #为了理解精确率和召唤率，需要对1类错误和2类错误有基本认识，我们知道类别为1的数据点常常被叫做正样本，类别为0（或者-1）
# # 的数据点常常叫做负样本。那么对一个特定的数据点地分类可能有4种结果，在统计假设检验中，家政也被称为1类错误，而假负被称为2类错误。

truly_a_positive=(y_true==1)
predicted_a_positive=(y_pred==1)
true_positive=np.sum(predicted_a_positive*truly_a_positive)
print(true_positive)

false_positive=np.sum((y_pred==1)*(y_true==0))
print(false_positive)

false_negative=np.sum((y_pred==0)*(y_true==1))
print(false_negative)

true_negative=np.sum((y_pred==0)*(y_true==0))
print(true_negative)

# 准确率，应该是真正的数量加上真负的数量（也就是得到的所有正确的结果数据），再除以数据点的总数量
print("len(y_true)=",len(y_true))
accuracy=(true_positive+true_negative)/len(y_true)
print(accuracy)

#用真正的数量除以所有被预测为正的结果数量得到精确率;
precision=true_positive/(true_positive+true_negative)
print(precision)

b=metrics.precision_score(y_true,y_pred)
print(b)

recall=true_positive/(true_positive+false_negative)
print(recall)

c=metrics.recall_score(y_true,y_pred)
print(c)