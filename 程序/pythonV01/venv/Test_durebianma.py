from sklearn.feature_extraction import  DictVectorizer

data=[
    {'name':'Alan Turing','born':1912,'died':1954},
    {'name':'Herbert A. Simon','born':1916,'died':2001},
    {'name':'Jacek Karpinski','born':1927,'died':2010},
    {'name':'J.C.R. Licklider','born':1915,'died':1990},
    {'name':'Marvin Minsky','born':1927,'died':2016}
]

vec=DictVectorizer(sparse=False,dtype=int)
out1=vec.fit_transform(data)
print(out1)
#
# out2=vec.get_feature_names()
# print(out2)
# 此方法的局限，如果特征的类别有很多可能的值，独热矩阵会产生非常大的数据矩阵。产生稀疏矩阵。
# scikit-learn提供了稀疏矩阵的及凑得表示方法，可以在第一次调用DictVectorizer时指定sparse=True来进行触发;
###########################################################################################
#进行紧凑的表示方法;
vec=DictVectorizer(sparse=True,dtype=int)
out1=vec.fit_transform(data)
print(out1)
