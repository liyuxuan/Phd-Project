import matplotlib.pyplot as plt
plt.style.use("ggplot")

# 加载数据信息内容;
data=[
    {"age":33,"sex":"F","BP":"high","cholesterol":"high","Na":0.66,"K":0.06,"drug":"A"},
    {"age":77,"sex":"F","BP":"high","cholesterol":"normal","Na":0.19,"K":0.03,"drug":"D"},
    {"age":88,"sex":"M","BP":"high","cholesterol":"normal","Na":0.80,"K":0.05,"drug":"B"}
]

# 列表中提取出"Drug"
target    =[d["drug"] for d in data]

# 列表中提取出"age","Na","K"
age       =[d["age"] for d in data]
sodium    =[d["Na"] for d in data]
potassium =[d["K"] for d in data]

target    =[ord(t)-65 for t in target]
# plt.subplot(231)表示2行，3列，第1个
plt.subplot(231)
plt.scatter(sodium,potassium,c=target,s=100)
plt.xlabel("sodium (Na)")
plt.ylabel("potassium (K)")

# ##############################################################
plt.subplot(232)
plt.scatter(age,potassium,c=target,s=100)
plt.xlabel("age")
plt.ylabel("potassium (K)")

# ##############################################################
plt.subplot(233)
plt.scatter(age,sodium,c=target,s=100)
plt.xlabel("age")
plt.ylabel("sodium (Na)")

plt.show()