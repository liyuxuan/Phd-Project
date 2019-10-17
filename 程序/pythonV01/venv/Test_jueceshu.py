data=[
    {"age":33,"sex":"F","BP":"high","cholesterol":"high","Na":0.66,"K":0.06,"drug":"A"},
    {"age":77,"sex":"F","BP":"high","cholesterol":"normal","Na":0.19,"K":0.03,"drug":"D"},
    {"age":88,"sex":"M","BP":"high","cholesterol":"normal","Na":0.80,"K":0.05,"drug":"B"}
]

target=[d["drug"] for d in data]

# print(target)

# 从数据中移除所有的"drug"

[d.pop("drug") for d in data];

# print(data)

import matplotlib.pyplot as plt
plt.style.use("ggplot")

age=[d["age"] for d in data]
print(age)

sodium      =[d["Na"] for d in data]
potassium   =[d["K"] for d in data]

# plt.scatter(sodium,potassium)
# plt.xlabel("sodium")
# plt.ylabel("potassium")
# plt.show()
###########################################################
#为标签选择较为合适的位置内容信息;

target=[ord(t)-65 for t in target]
print(target)

plt.subplot(221)
plt.scatter(sodium,potassium,c=target,s=100)

plt.xlabel("sodium (Na)")
plt.ylabel("potassium (K)")

plt.subplot(222)
plt.scatter(age,potassium,c=target,s=100)

plt.xlabel("age")
plt.ylabel("potassium (K)")

plt.subplot(223)
plt.scatter(age,sodium,c=target,s=100)

plt.xlabel("age")
plt.ylabel("sodium (Na)")

plt.show()




