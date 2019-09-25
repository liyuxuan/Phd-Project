from sklearn import datasets

# mnist=datasets.fetch_mldata("MNIST original")
mnist=datasets.fetch_mldata("MNIST original")
print(mnist.data.shape)
print(mnist.target.shape)

# minst数据集中包含了手写数字的70000个样例（28x28像素的图像，标记为0到9）。数据和标签存储于2个不同的容器中，并且每张图片只有一个标签;

import numpy as np
array=np.unique(mnist.target)
print(array)