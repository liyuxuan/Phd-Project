import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use("ggplot")

np.random.seed(42)

single_data_point=np.random.randint(0,100,2)
print(single_data_point)

single_label=np.random.randint(0,2)
print(single_label)

def generate_data(num_samples,num_features=2):
    data_size=(num_samples,num_features)
    data=np.random.randint(0,100,data_size)
    label_size=(num_samples,1)
    labels=np.random.randint(0,2,size=label_size)
    return data.astype(np.float32),labels

#OpenCV对于数据类型具有过分的要求，因此确保总是把数据点的类型转换为np.float32
train_data,labels=generate_data(11)
# print(train_data,train_data.dtype,labels)
print(train_data[0],labels[0])

plt.plot(train_data[0,0],train_data[0,1],"sb")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")

def plot_data(all_blue,all_red):
    plt.scatter(all_blue[:,0],all_blue[:,1],c="b",marker="s",s=180)
    plt.scatter(all_red[:,0],all_red[:,1],c="r",marker="^",s=180)
    plt.xlabel("x coordinate (feature 1)")
    plt.ylabel("y coordinate (feature 2)")

labels.ravel()==0
print(labels.ravel(),labels.ravel().dtype)

blue=train_data[labels.ravel()==0]

red=train_data[labels.ravel()==1]

plot_data(blue,red)
plt.savefig("figures/a.png")
# plt.show()

# 训练分类器;
knn=cv2.ml.KNearest_create()
# print(knn)
a=knn.train(train_data,cv2.ml.ROW_SAMPLE,labels)
print(a)

newcomer,_=generate_data(1)
print(newcomer)
#根据generate_data函数，生成新的数据点，把新数据点当做只有一个数据的数据集。

plot_data(blue,red)
plt.plot(newcomer[0,0],newcomer[0,1],"go",markersize=14);
#在plt.plot之后加上一个分号抑制输出;

# plt.show()


ret,results,neighbor,dist=knn.findNearest(newcomer,1)
print("Predicted label:\t",results)
print("Neighbor's label",neighbor)
print("Distance to neighbor:\t",dist)

