sample=["feature engineering","feature selection","feature extraction"]

# # 对文本内容进行编码的常见方式为使用单词计数。对于每个短语，仅仅计算其中的每个单
# # 词的出出现的次数。在scikit-learn中，使用CountVectorizer可以完成。
# # 功能与DictVectorizer相似。
#
# from sklearn.feature_extraction.text import CountVectorizer
# vec=CountVectorizer()
# X=vec.fit_transform(sample)
# print(X)

# #操作默认保存为稀疏矩阵。如果手动检测他，首先需要转换成一个常规数组;
# print(X.toarray())
#
# out1=vec.get_feature_names()
# print(out1)

from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer()
X=vec.fit_transform(sample)
out2=X.toarray()
print(out2)

print(vec.get_feature_names())
#第三列数据较大因为对应的是出现次数最多的feature

