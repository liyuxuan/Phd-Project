from sklearn import tree
dtc     =   tree.DecisionTreeClassifier()
out1=dtc.feature_importances_
print(out1)